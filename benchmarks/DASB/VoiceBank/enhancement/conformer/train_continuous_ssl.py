#!/usr/bin/env/python

"""Recipe for training a transformer-based speech enhancement system using continuous SSL audio representations.

To run this recipe:
> python train_continuous_ssl.py hparams/<path-to-config>.yaml

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


# To use in configuration files
def len_(SSL_layers, embedding_dim):
    return len(SSL_layers) * embedding_dim


class Enhancement(sb.Brain):
    @torch.no_grad()
    def sig_to_embs(self, sig, lens):
        # sig: [B, T]
        self.hparams.ssl_model.to(self.device).eval()
        embs = self.hparams.ssl_model(sig, lens)[
            self.hparams.SSL_layers
        ]  # [K, B, N, H]
        embs = embs.movedim(0, -2)  # [B, N, K, H]
        return embs

    @torch.no_grad()
    def embs_to_sig(self, embs):
        # embs: [B, N, K, H]
        self.hparams.ssl_vocoder.device = self.device
        self.hparams.ssl_vocoder.to(self.device).eval()

        # Handle missing codebooks
        all_layer_ids = [1, 3, 7, 12, 18, 23]
        if len(self.hparams.SSL_layers) < len(all_layer_ids):
            offset_idxes = [
                all_layer_ids.index(x) for x in self.hparams.SSL_layers
            ]
            full_embs = torch.zeros(
                *embs.shape[:2],
                len(all_layer_ids),
                embs.shape[-1],
                dtype=embs.dtype,
                device=self.device,
            )
            for i, idx in enumerate(offset_idxes):
                full_embs[..., idx, :] = embs[..., i, :]
            embs = full_embs

        self.hparams.ssl_vocoder.tokenize = False
        sig = self.hparams.ssl_vocoder(embs)[:, 0]  # [B, T]
        return sig

    def compute_forward(self, batch, stage):
        """Forward pass."""
        batch = batch.to(self.device)
        in_sig, in_lens = batch.in_sig  # [B, T]
        out_sig, out_lens = batch.out_sig  # [B, T]

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            in_sig, in_lens = self.hparams.augmentation(in_sig, in_lens)

        # Extract features (cache them at first epoch if augmentation is disabled)
        key = tuple(sorted(batch.id))
        try:
            in_embs, out_embs = _CACHE[key]
            in_embs = in_embs.to(self.device)
            out_embs = out_embs.to(self.device)
        except KeyError:
            assert (in_lens == out_lens).all()
            sig = torch.cat([in_sig, out_sig])  # [B2, T]
            lens = torch.cat([in_lens, out_lens])  # [B2, T]
            embs = self.sig_to_embs(sig, lens)  # [B2, N, K, H]
            in_embs, out_embs = embs.split(
                [len(in_sig), len(out_sig)]
            )  # [B, N, K, H], [B, N, K, H]
            out_embs = out_embs.reshape(
                len(in_sig),
                -1,
                self.hparams.num_codebooks,
                self.hparams.embedding_dim,
            )  # [B, N, K, H]
            if self.hparams.use_cache and (not self.hparams.augment):
                _CACHE[key] = in_embs.cpu(), out_embs.cpu()

        # Avoid in-place modification from attention
        in_embs = in_embs.clone()

        # Forward attention
        att_w = self.modules.attention_mlp(in_embs)  # [B, N, K, 1]
        in_embs = torch.matmul(att_w.transpose(2, -1), in_embs).squeeze(
            -2
        )  # [B, N, H]

        # Forward encoder
        hyp_embs = self.modules.encoder.encode(in_embs, in_lens)  # [B, N, H]

        # Forward head
        hyp_embs = self.modules.head(hyp_embs).reshape(
            len(hyp_embs),
            -1,
            self.hparams.num_codebooks,
            self.hparams.embedding_dim,
        )  # [B, N, K, H]

        return hyp_embs, out_embs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        hyp_embs, out_embs = predictions  # [B, N, K, H], [B, N, K, H]

        IDs = batch.id
        in_sig, _ = batch.in_sig
        out_sig, out_lens = batch.out_sig

        # Reconstruction loss
        loss = self.hparams.rec_loss(
            hyp_embs.flatten(start_dim=1, end_dim=-2),  # [B, NK, H]
            out_embs.flatten(start_dim=1, end_dim=-2),  # [B, NK, H]
            length=out_lens,
        )

        # Vocode
        if stage == sb.Stage.TEST and self.hparams.compute_metrics:
            self.vocode(IDs, in_sig, out_sig, hyp_embs, out_embs, out_lens)

        return loss

    @torch.no_grad()
    def vocode(self, IDs, in_sig, out_sig, hyp_embs, out_embs, lens):
        hyp_sig = self.embs_to_sig(hyp_embs)  # [B, T]
        rec_sig = self.embs_to_sig(out_embs)  # [B, T]

        # Adjust length
        if out_sig.shape[-1] > hyp_sig.shape[-1]:
            pad = [0, out_sig.shape[-1] - hyp_sig.shape[-1]]
            hyp_sig = torch.nn.functional.pad(
                hyp_sig, pad, mode="replicate"
            )  # [B, T_out]
            rec_sig = torch.nn.functional.pad(
                rec_sig, pad, mode="replicate"
            )  # [B, T_out]
        elif out_sig.shape[-1] < hyp_sig.shape[-1]:
            hyp_sig = hyp_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]
            rec_sig = rec_sig.narrow(-1, 0, out_sig.shape[-1])  # [B, T_out]

        self.dnsmos_metric.append(IDs, hyp_sig, lens)
        self.rec_dnsmos_metric.append(IDs, rec_sig, lens)
        self.ref_dnsmos_metric.append(IDs, out_sig, lens)
        self.dwer_metric.append(IDs, hyp_sig, out_sig, lens)
        self.wavlm_sim_metric.append(IDs, hyp_sig, out_sig, lens)
        self.ecapatdnn_sim_metric.append(IDs, hyp_sig, out_sig, lens)

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

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            _, lr = self.hparams.scheduler(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]},
                min_keys=["loss"],
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
    from voicebank_prepare import prepare_voicebank as prepare_data

    prepare_data_kwargs = {
        "data_folder": hparams["data_folder"],
        "save_folder": hparams["save_folder"],
        "splits": hparams["splits"],
        "num_valid_speakers": hparams["num_valid_speakers"],
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

    # Log number of parameters/buffers
    ssl_params = sum(
        [x.numel() for x in hparams["ssl_model"].state_dict().values()]
        + [x.numel() for x in hparams["ssl_vocoder"].state_dict().values()]
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
            f"SSL parameters/buffers (M)": f"{ssl_params / 1e6:.2f}",
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Trainer initialization
    brain = Enhancement(
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
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
