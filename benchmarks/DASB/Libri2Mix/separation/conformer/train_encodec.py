#!/usr/bin/env/python

"""Recipe for training a transformer-based speech separation system using EnCodec audio representations.

To run this recipe:
> python train_encodec.py hparams/<path-to-config>.yaml

Authors
 * Luca Della Libera 2024
"""

import os
import sys
import warnings

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import write_audio
from speechbrain.utils.distributed import if_main_process, run_on_main


_CACHE = {}


class Separation(sb.Brain):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        # sig: [B, T]
        self.hparams.codec.to(self.device).eval()
        toks, _ = self.hparams.codec.encode(sig, lens)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        # toks: [B, N, K]
        self.hparams.codec.to(self.device).eval()
        sig = self.hparams.codec.decode(toks)[:, 0]  # [B, T]
        return sig

    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        in_sig, in_lens = batch.in_sig  # [B, T]
        out_sig, out_lens = batch.out_sig  # [B, ST]

        # Unflatten
        out_sig = out_sig.reshape(
            len(out_sig), self.hparams.num_speakers, -1
        ).flatten(
            end_dim=-2
        )  # [BS, T]
        batch.out_sig = out_sig, out_lens

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            in_sig, in_lens = self.hparams.augmentation(in_sig, in_lens)

        # Extract tokens (cache them at first epoch if augmentation is disabled)
        key = tuple(sorted(batch.id))
        try:
            in_toks, out_toks = _CACHE[key]
            in_toks = in_toks.to(self.device)
            out_toks = out_toks.to(self.device)
        except KeyError:
            assert (in_lens == out_lens).all()
            sig = torch.cat([in_sig, out_sig])  # [B(1 + S), T]
            lens = torch.cat(
                [
                    in_lens,
                    out_lens.repeat_interleave(self.hparams.num_speakers),
                ]
            )  # [B(1 + S), T]
            toks = self.sig_to_toks(sig, lens)  # [B(1 + S), N, K]
            in_toks, out_toks = toks.split(
                [len(in_sig), len(out_sig)]
            )  # [B, N, K], [BS, N, K]
            out_toks = out_toks.reshape(
                len(in_sig),
                self.hparams.num_speakers,
                -1,
                self.hparams.num_codebooks,
            ).movedim(
                -2, -3
            )  # [B, N, S, K]
            if self.hparams.use_cache and (not self.hparams.augment):
                _CACHE[key] = in_toks.cpu(), out_toks.cpu()

        # Avoid in-place modification from embedding layer
        in_toks = in_toks.clone()

        # Forward embedding + attention
        in_embs = self.modules.embedding(in_toks)  # [B, N, K, H]
        att_w = self.modules.attention_mlp(in_embs)  # [B, N, K, 1]
        in_embs = torch.matmul(att_w.transpose(2, -1), in_embs).squeeze(
            -2
        )  # [B, N, H]

        # Forward encoder
        hyp_embs = self.modules.encoder.encode(in_embs, in_lens)  # [B, N, H]

        # Forward head
        log_probs = (
            self.modules.head(hyp_embs)
            .reshape(
                len(hyp_embs),
                -1,
                self.hparams.num_speakers,
                self.hparams.num_codebooks,
                self.hparams.vocab_size,
            )
            .log_softmax(dim=-1)
        )  # [B, N, S, K, C]

        return log_probs, out_toks

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        log_probs, out_toks = predictions  # [B, N, S, K, C], [B, N, S, K]

        IDs = batch.id
        in_sig, _ = batch.in_sig
        out_sig, out_lens = batch.out_sig

        if not self.hparams.use_pit:
            # Cross-entropy loss
            loss = self.hparams.ce_loss(
                log_probs.flatten(start_dim=1, end_dim=3),  # [B, NSK, C]
                out_toks.flatten(start_dim=1),  # [B, NSK]
                length=out_lens,
            )
        else:
            # Permutation invariant training
            from speechbrain.nnet.losses import PitWrapper

            def base_loss(preds, targets):
                # preds: [N, K, C, S, S]
                # targets: [N, K, S, S]
                preds = preds.permute(3, 4, 0, 1, 2)  # [S, S, N, K, C]
                targets = targets.permute(2, 3, 0, 1)  # [S, S, N, K]
                loss = self.hparams.ce_loss(
                    preds.flatten(end_dim=-2),
                    targets.flatten(),
                    reduction="none",
                )  # [SSNK]
                loss = loss.reshape_as(targets)
                loss = loss.permute(2, 3, 0, 1)  # [N, K, S, S]
                return loss

            log_probs = log_probs.movedim(2, -1)  # [B, N, K, C, S]
            out_toks = out_toks.movedim(2, -1)  # [B, N, K, S]
            pit_loss = PitWrapper(base_loss)
            log_probs_list = [
                x[: int(l * log_probs.shape[1])]
                for x, l in zip(log_probs, out_lens)
            ]
            out_toks_list = [
                x[: int(l * out_toks.shape[1])]
                for x, l in zip(out_toks, out_lens)
            ]
            loss, perm = pit_loss(log_probs_list, out_toks_list)
            loss = loss.mean()
            log_probs = pit_loss.reorder_tensor(log_probs, perm)
            log_probs = log_probs.movedim(-1, 2)  # [B, N, S, K, C]
            out_toks = out_toks.movedim(-1, 2)  # [B, N, S, K]

        # Compute TER
        if stage != sb.Stage.TRAIN:
            self.ter_metric.append(
                IDs,
                log_probs.flatten(start_dim=1, end_dim=3),
                out_toks.flatten(start_dim=1),
                out_lens,
            )

        # Vocode
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            hyp_toks = log_probs.argmax(dim=-1)  # [B, N, S, K]
            self.vocode(IDs, in_sig, out_sig, hyp_toks, out_toks, out_lens)

        return loss

    @torch.no_grad()
    def vocode(self, IDs, in_sig, out_sig, hyp_toks, out_toks, lens):
        hyp_toks = hyp_toks.movedim(-2, -3).contiguous()  # [B, S, N, K]
        out_toks = out_toks.movedim(-2, -3).contiguous()  # [B, S, N, K]

        hyp_sig = self.toks_to_sig(
            hyp_toks.flatten(end_dim=1)  # [BS, N, K]
        )  # [BS, T]
        rec_sig = self.toks_to_sig(
            out_toks.flatten(end_dim=1)  # [BS, N, K]
        )  # [BS, T]

        # Adjust length
        if out_sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, out_sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(
                hyp_sig, pad, mode="replicate"
            )  # [BS, T_out]
            rec_sig = torch.nn.functional.pad(
                rec_sig, pad, mode="replicate"
            )  # [BS, T_out]
        elif out_sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, out_sig.shape[-1])  # [BS, T_out]
            rec_sig = rec_sig.narrow(-1, 0, out_sig.shape[-1])  # [BS, T_out]

        all_IDs = [
            f"{x}_{i}" for x in IDs for i in range(self.hparams.num_speakers)
        ]
        all_lens = lens.repeat_interleave(self.hparams.num_speakers)
        self.dnsmos_metric.append(all_IDs, hyp_sig, all_lens)
        self.rec_dnsmos_metric.append(all_IDs, rec_sig, all_lens)
        self.ref_dnsmos_metric.append(all_IDs, out_sig, all_lens)
        self.dwer_metric.append(all_IDs, hyp_sig, out_sig, all_lens)
        self.wavlm_sim_metric.append(all_IDs, hyp_sig, out_sig, all_lens)
        self.ecapatdnn_sim_metric.append(all_IDs, hyp_sig, out_sig, all_lens)

        hyp_sig = hyp_sig.reshape(len(hyp_toks), -1)  # [B, ST_out]
        rec_sig = rec_sig.reshape(len(hyp_toks), -1)  # [B, ST_out]
        out_sig = out_sig.reshape(len(hyp_toks), -1)  # [B, ST_out]

        if self.hparams.save_audios:
            save_folder = os.path.join(self.hparams.output_folder, "audios")
            os.makedirs(save_folder, exist_ok=True)
            for i in range(len(IDs)):
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_hyp.wav"),
                    hyp_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_rec.wav"),
                    rec_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_ref.wav"),
                    out_sig[i].cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{IDs[i]}_in.wav"),
                    in_sig[i].cpu(),
                    self.hparams.sample_rate,
                )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        super().on_stage_start(stage, epoch)
        if stage != sb.Stage.TRAIN:
            self.ter_metric = self.hparams.ter_computer()
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.dnsmos_metric = self.hparams.dnsmos_computer()
            self.rec_dnsmos_metric = self.hparams.dnsmos_computer()
            self.ref_dnsmos_metric = self.hparams.dnsmos_computer()
            self.dwer_metric = self.hparams.dwer_computer()
            self.wavlm_sim_metric = self.hparams.wavlm_sim_computer()
            self.ecapatdnn_sim_metric = self.hparams.ecapatdnn_sim_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["TER"] = self.ter_metric.summarize("average") * 100

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            _, lr = self.hparams.scheduler(stage_stats["TER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"TER": stage_stats["TER"]},
                min_keys=["TER"],
                num_to_keep=self.hparams.keep_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            if self.hparams.compute_metrics:
                stage_stats["DNSMOS"] = self.dnsmos_metric.summarize("average")
                stage_stats["RecDNSMOS"] = self.rec_dnsmos_metric.summarize(
                    "average"
                )
                stage_stats["RefDNSMOS"] = self.ref_dnsmos_metric.summarize(
                    "average"
                )
                stage_stats["dWER"] = self.dwer_metric.summarize("error_rate")
                stage_stats["WavLMSim"] = self.wavlm_sim_metric.summarize(
                    "average"
                )
                stage_stats[
                    "ECAPATDNNSim"
                ] = self.ecapatdnn_sim_metric.summarize("average")
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                # Save dWER
                if self.hparams.compute_metrics:
                    with open(self.hparams.dwer_file, "w") as w:
                        self.dwer_metric.write_stats(w)


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Filter warnings
    warnings.filterwarnings("once")
    warnings.filterwarnings("ignore", module="torch")

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset preparation
    from librimix_prepare import prepare_librimix as prepare_data

    prepare_data_kwargs = {
        "data_folder": hparams["data_folder"],
        "save_folder": hparams["save_folder"],
        "splits": hparams["splits"],
        "num_speakers": hparams["num_speakers"],
        "add_noise": hparams["add_noise"],
        "version": hparams["version"],
    }

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(prepare_data, kwargs=prepare_data_kwargs)

    # Create the datasets objects
    from utils import dataio_prepare

    train_data, valid_data, test_data = dataio_prepare(
        debug=run_opts.get("debug", False), **hparams
    )

    # Pretrain the specified modules
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        run_on_main(hparams["pretrainer"].load_collected)

    # Use pretrained embeddings
    if hparams["pretrain_embedding"]:
        embs = hparams["codec"].vocabulary.reshape(-1, hparams["embedding_dim"])
        hparams["embedding"].embedding.weight.data.copy_(embs)

    # Log number of parameters/buffers
    codec_params = sum(
        [x.numel() for x in hparams["codec"].state_dict().values()]
    )
    model_params = sum(
        [
            x.numel()
            for module in hparams["modules"].values()
            for x in module.state_dict().values()
        ]
    )
    hparams["train_logger"].log_stats(
        stats_meta={
            f"Codec parameters/buffers (M)": f"{codec_params / 1e6:.2f}",
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Trainer initialization
    brain = Separation(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.hparams.dwer_file = os.path.join(hparams["output_folder"], "dwer.txt")
    brain.evaluate(
        test_data,
        min_key="TER",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
