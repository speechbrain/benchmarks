#!/usr/bin/env/python

"""Recipe for training an encoder-only CRDNN-based speech separation system using
discrete audio representations (see https://arxiv.org/abs/2312.09747).

The model is trained via cross-entropy loss applied to each timestep using discrete HuBERT audio
representations (see https://arxiv.org/abs/2106.07447, https://arxiv.org/abs/2309.07377).

The neural network architecture is inspired by:
https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py#L27

To run this recipe:
> python train_discrete_hubert.py hparams/train_discrete_hubert.yaml

Authors
 * Luca Della Libera 2023
"""

import csv
import itertools
import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import if_main_process, run_on_main


class Separation(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        current_epoch = self.hparams.epoch_counter.current

        batch = batch.to(self.device)
        in_sig, in_sig_lens = batch.in_sig
        out_sig, out_sig_lens = batch.out_sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            in_sig, in_sig_lens = self.hparams.augmentation(in_sig, in_sig_lens)

        # Unflatten
        out_sig = out_sig.reshape(
            len(batch), self.hparams.num_speakers, -1
        )  # [B, S, T]

        # Stack along batch dimension
        out_sig = out_sig.reshape(-1, out_sig.shape[-1])  # [B * S, T]

        # Extract audio tokens
        assert (in_sig_lens == out_sig_lens).all()
        sig = torch.cat([in_sig, out_sig])
        lens = torch.cat(
            [
                in_sig_lens,
                out_sig_lens[:, None]
                .expand(-1, self.hparams.num_speakers)
                .flatten(),
            ]
        )
        with torch.no_grad():
            self.hparams.codec.to(self.device).eval()
            tokens = self.hparams.codec(sig, lens)[1][..., None]
        in_tokens = tokens[: len(tokens) // (self.hparams.num_speakers + 1)]
        out_tokens = tokens[
            len(tokens) // (self.hparams.num_speakers + 1) :
        ]  # [B * S, N, K]
        out_tokens = out_tokens.reshape(
            len(batch),
            self.hparams.num_speakers,
            -1,
            self.hparams.num_codebooks,
        )  # [B, S, N, K]
        out_tokens = out_tokens.movedim(-3, -1).reshape(
            len(batch),
            -1,
            self.hparams.num_codebooks * self.hparams.num_speakers,
        )  # [B, N, K * S]
        batch.out_tokens = out_tokens, out_sig_lens

        # Forward embedding layer (one for each codebook/speaker)
        in_tokens += torch.arange(
            0,
            self.hparams.num_codebooks * self.hparams.vocab_size,
            self.hparams.vocab_size,
            device=self.device,
        )  # Offset to select embeddings from the correct codebook
        in_embs = (
            self.modules.embedding(in_tokens)
            .reshape(
                len(batch),
                -1,
                self.hparams.embedding_dim,
                self.hparams.num_codebooks,
            )
            .sum(dim=-1)
        )

        # Forward encoder
        enc_out = self.modules.encoder(in_embs)

        # Compute cross-entropy logits (one for each codebook/speaker)
        ce_logits = self.modules.ce_head(enc_out).reshape(
            len(batch),
            -1,
            self.hparams.num_codebooks * self.hparams.num_speakers,
            self.hparams.vocab_size,
        )

        # Compute outputs
        hyps = None
        if (
            stage == sb.Stage.TEST
            # During validation, run decoding only every valid_search_freq epochs to speed up training
            or (
                stage == sb.Stage.VALID
                and current_epoch % self.hparams.valid_search_freq == 0
            )
        ):
            hyps = ce_logits.argmax(dim=-1).flatten(start_dim=-2).tolist()

            # Remove padding (output length is equal to input length)
            min_length = self.hparams.num_codebooks * self.hparams.num_speakers
            hyps = [
                hyp[: min_length * int(len(hyp) * rel_length / min_length)]
                for hyp, rel_length in zip(hyps, in_sig_lens)
            ]

        return ce_logits, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        ce_logits, hyps = predictions

        IDs = batch.id
        out_tokens, out_tokens_lens = batch.out_tokens

        if not self.hparams.use_pit:
            # Cross-entropy loss
            loss = self.hparams.ce_loss(
                ce_logits.log_softmax(dim=-1).flatten(start_dim=-3, end_dim=-2),
                out_tokens.flatten(start_dim=-2),
                length=out_tokens_lens,
            )
        else:
            # Apply permutations
            batch_size = len(out_tokens)
            num_perms = len(self.perm_matrix)
            num_heads = self.hparams.num_codebooks * self.hparams.num_speakers
            perm_out_tokens = (
                (
                    out_tokens.reshape(-1, self.hparams.num_speakers)
                    .expand(num_perms, -1, -1)
                    .type(self.perm_matrix.dtype)
                    @ self.perm_matrix
                )
                .reshape(num_perms, batch_size, -1, num_heads,)
                .long()
            )

            # Cross-entropy loss
            ce_logits = ce_logits.expand(num_perms, -1, -1, -1, -1).reshape(
                num_perms * batch_size, -1, num_heads, self.hparams.vocab_size
            )
            perm_out_tokens = perm_out_tokens.reshape(
                num_perms * batch_size, -1, num_heads
            )
            loss = self.hparams.ce_loss(
                ce_logits.log_softmax(dim=-1).flatten(start_dim=-3, end_dim=-2),
                perm_out_tokens.flatten(start_dim=-2),
                length=out_tokens_lens.expand(num_perms, -1).flatten(),
                reduction="batch",
            ).reshape(num_perms, batch_size)
            loss, idxes = loss.min(dim=0)
            loss = loss.mean()

            # Select tokens corresponding to the permutation with minimum loss
            out_tokens = perm_out_tokens[
                idxes * batch_size
                + torch.arange(0, batch_size, device=self.device)
            ]

        if hyps is not None:
            targets = out_tokens.flatten(start_dim=-2).tolist()

            # Remove padding
            min_length = self.hparams.num_codebooks * self.hparams.num_speakers
            targets = [
                target[
                    : min_length * int(len(target) * rel_length / min_length)
                ]
                for target, rel_length in zip(targets, out_tokens_lens)
            ]

            # Compute TER
            self.ter_metric.append(IDs, hyps, targets)

        return loss

    def evaluate(self, test_set, *args, **kwargs):
        """Evaluation loop."""
        self.test_set = test_set
        return super().evaluate(test_set, *args, **kwargs)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch."""
        if stage != sb.Stage.TRAIN:
            self.ter_metric = self.hparams.ter_computer()
        if self.hparams.use_pit:
            # Batch of permutation matrices (one for each of the factorial(num_speakers) permutations)
            perms = itertools.permutations(range(self.hparams.num_speakers))
            self.perm_matrix = torch.stack(
                [
                    torch.eye(self.hparams.num_speakers, device=self.device)[
                        torch.as_tensor(perm)
                    ]
                    for perm in perms
                ]
            )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of each epoch."""
        # Compute/store important stats
        current_epoch = self.hparams.epoch_counter.current
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_freq == 0
        ) or stage == sb.Stage.TEST:
            stage_stats["TER"] = self.ter_metric.summarize("error_rate")

        # Perform end-of-iteration operations, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.scheduler.hyperparam_value
            if (
                self.hparams.enable_scheduler
                and current_epoch % self.hparams.valid_search_freq == 0
            ):
                _, lr = self.hparams.scheduler(stage_stats["TER"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, lr)
            steps = self.optimizer_step
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": steps},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if current_epoch % self.hparams.valid_search_freq == 0:
                if if_main_process():
                    self.checkpointer.save_and_keep_only(
                        meta={"TER": stage_stats["TER"]},
                        min_keys=["TER"],
                        num_to_keep=self.hparams.keep_checkpoints,
                    )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": current_epoch},
                test_stats=stage_stats,
            )

            if if_main_process():
                # Save TER
                with open(self.hparams.ter_file, "w") as w:
                    self.ter_metric.write_stats(w)

                # Vocode
                if self.hparams.vocode:
                    self.vocode()

    def vocode(self):
        from speechbrain.nnet.loss.si_snr_loss import si_snr_loss
        from tqdm import tqdm

        from cos_sim import CosSim
        from dnsmos import DNSMOS
        from dwer import DWER

        IDs = []
        sisnrs = []
        dnsmoses = []
        rec_dnsmoses = []
        ref_dnsmoses = []
        cos_sims = []
        dwers = []
        texts = []
        ref_texts = []
        for score in tqdm(
            self.ter_metric.scores,
            dynamic_ncols=True,
            total=len(self.ter_metric.scores),
        ):
            ID, hyp_tokens, rec_tokens = (
                score["key"],
                score["hyp_tokens"],
                score["ref_tokens"],
            )

            # Decode
            hyp_tokens = (
                torch.as_tensor(hyp_tokens, device=self.device)
                .reshape(
                    -1, self.hparams.num_codebooks, self.hparams.num_speakers,
                )
                .movedim(-1, 0)
            )
            rec_tokens = (
                torch.as_tensor(rec_tokens, device=self.device)
                .reshape(
                    -1, self.hparams.num_codebooks, self.hparams.num_speakers,
                )
                .movedim(-1, 0)
            )

            with torch.no_grad():
                self.hparams.vocoder.device = self.device
                self.hparams.vocoder.to(self.device).eval()
                hyp_sig = self.hparams.vocoder.decode_batch(hyp_tokens[..., 0])[
                    :, 0
                ].flatten()
                rec_sig = self.hparams.vocoder.decode_batch(rec_tokens[..., 0])[
                    :, 0
                ].flatten()
            ref_sig = self.test_set[self.test_set.data_ids.index(ID)][
                "out_sig"
            ].to(
                self.device
            )  # Original output signal (resampled and flattened)
            in_sig = self.test_set[self.test_set.data_ids.index(ID)][
                "in_sig"
            ]  # Original input signal (resampled)

            if self.hparams.save_audios:
                save_folder = os.path.join(self.hparams.output_folder, "audios")
                os.makedirs(save_folder, exist_ok=True)
                torchaudio.save(
                    os.path.join(save_folder, f"{ID}_hyp.wav"),
                    hyp_sig[None].cpu(),
                    self.hparams.sample_rate,
                )
                torchaudio.save(
                    os.path.join(save_folder, f"{ID}_rec.wav"),
                    rec_sig[None].cpu(),
                    self.hparams.sample_rate,
                )
                torchaudio.save(
                    os.path.join(save_folder, f"{ID}_ref.wav"),
                    ref_sig[None].cpu(),
                    self.hparams.sample_rate,
                )
                torchaudio.save(
                    os.path.join(save_folder, f"{ID}_in.wav"),
                    in_sig[None].cpu(),
                    self.hparams.sample_rate,
                )

            # Compute metrics (average over speakers)
            spk_sisnrs = []
            spk_dnsmoses = []
            spk_rec_dnsmoses = []
            spk_ref_dnsmoses = []
            spk_cos_sims = []
            spk_dwers = []
            spk_texts = []
            spk_ref_texts = []

            spk_hyp_sig_length = len(hyp_sig) // self.hparams.num_speakers
            spk_rec_sig_length = len(rec_sig) // self.hparams.num_speakers
            spk_ref_sig_length = len(ref_sig) // self.hparams.num_speakers

            for i in range(self.hparams.num_speakers):
                spk_hyp_sig = hyp_sig[
                    i * spk_hyp_sig_length : (i + 1) * spk_hyp_sig_length
                ]
                spk_rec_sig = rec_sig[
                    i * spk_rec_sig_length : (i + 1) * spk_rec_sig_length
                ]
                spk_ref_sig = ref_sig[
                    i * spk_ref_sig_length : (i + 1) * spk_ref_sig_length
                ]

                spk_min_length = min(len(spk_hyp_sig), len(spk_ref_sig))
                spk_sisnr = -si_snr_loss(
                    spk_hyp_sig[None, :spk_min_length],
                    spk_ref_sig[None, :spk_min_length],
                    torch.as_tensor([1.0], device=self.device),
                ).item()
                spk_dnsmos = DNSMOS(spk_hyp_sig, self.hparams.sample_rate)
                spk_rec_dnsmos = DNSMOS(spk_rec_sig, self.hparams.sample_rate)
                spk_ref_dnsmos = DNSMOS(spk_ref_sig, self.hparams.sample_rate)
                spk_cos_sim = CosSim(
                    spk_hyp_sig, spk_ref_sig, self.hparams.sample_rate
                )
                spk_dwer, spk_text, spk_ref_text = DWER(
                    spk_hyp_sig, spk_ref_sig, self.hparams.sample_rate
                )

                spk_sisnrs.append(spk_sisnr)
                spk_dnsmoses.append(spk_dnsmos)
                spk_rec_dnsmoses.append(spk_rec_dnsmos)
                spk_ref_dnsmoses.append(spk_ref_dnsmos)
                spk_cos_sims.append(spk_cos_sim)
                spk_dwers.append(spk_dwer)
                spk_texts.append(spk_text)
                spk_ref_texts.append(spk_ref_text)

            sisnr = sum(spk_sisnrs) / len(spk_sisnrs)
            dnsmos = sum(spk_dnsmoses) / len(spk_dnsmoses)
            rec_dnsmos = sum(spk_rec_dnsmoses) / len(spk_rec_dnsmoses)
            ref_dnsmos = sum(spk_ref_dnsmoses) / len(spk_ref_dnsmoses)
            cos_sim = sum(spk_cos_sims) / len(spk_cos_sims)
            dwer = sum(spk_dwers) / len(spk_dwers)
            text = " || ".join(spk_texts)
            ref_text = " || ".join(spk_ref_texts)

            IDs.append(ID)
            sisnrs.append(sisnr)
            dnsmoses.append(dnsmos)
            rec_dnsmoses.append(rec_dnsmos)
            ref_dnsmoses.append(ref_dnsmos)
            cos_sims.append(cos_sim)
            dwers.append(dwer)
            texts.append(text)
            ref_texts.append(ref_text)

        headers = [
            "ID",
            "SI-SNR",
            "DNSMOS",
            "RecDNSMOS",
            "RefDNSMOS",
            "CosSim",
            "dWER",
            "text",
            "ref_text",
        ]
        with open(
            os.path.join(self.hparams.output_folder, "metrics.csv"),
            "w",
            encoding="utf-8",
        ) as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()

            for entry in zip(
                IDs,
                sisnrs,
                dnsmoses,
                rec_dnsmoses,
                ref_dnsmoses,
                cos_sims,
                dwers,
                texts,
                ref_texts,
            ):
                entry = dict(zip(headers, entry))
                csv_writer.writerow(entry)

            entry = dict(
                zip(
                    headers,
                    [
                        "Average",
                        sum(sisnrs) / len(sisnrs),
                        sum(dnsmoses) / len(dnsmoses),
                        sum(rec_dnsmoses) / len(rec_dnsmoses),
                        sum(ref_dnsmoses) / len(ref_dnsmoses),
                        sum(cos_sims) / len(cos_sims),
                        sum(dwers) / len(dwers),
                        "",
                        "",
                    ],
                )
            )
            csv_writer.writerow(entry)
            self.hparams.train_logger.log_stats(
                stats_meta={k: v for k, v in list(entry.items())[1:-2]},
            )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_max_value={"duration": hparams["train_remove_if_longer"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not run_opts.get("debug", False),
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not run_opts.get("debug", False),
        key_max_value={"duration": hparams["test_remove_if_longer"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline
    takes = (
        ["mix_clean_wav"]
        + [f"src{i}_wav" for i in range(1, hparams["num_speakers"] + 1)]
        + (["noise_wav"] if hparams["add_noise"] else [])
    )
    provides = ["in_sig", "out_sig"]

    def audio_pipeline(mix_wav, *src_and_noise_wavs):
        src_wavs = src_and_noise_wavs[: hparams["num_speakers"]]

        # Clean signals
        clean_sigs = []
        for src_wav in src_wavs:
            clean_sig, sample_rate = torchaudio.load(src_wav)
            clean_sigs.append(clean_sig)
        clean_sigs = torch.cat(clean_sigs)  # [S, T]

        # Mixed signal
        mix_sig, sample_rate = torchaudio.load(mix_wav)
        mix_sig = mix_sig[0]  # [T]

        if hparams["add_noise"]:
            # Noise signal
            noise_wav = src_and_noise_wavs[-1]
            noise_sig, sample_rate = torchaudio.load(noise_wav)
            noise_sig = noise_sig[0]  # [T]

            # Mixing with given noise gain
            noise_gain = -hparams["snr"]
            mix_sig_power = (mix_sig ** 2).mean()
            ratio = 10 ** (
                noise_gain / 10
            )  # ratio = noise_sig_power / mix_sig_power
            desired_noise_sig_power = ratio * mix_sig_power
            noise_sig_power = (noise_sig ** 2).mean()
            gain = (desired_noise_sig_power / noise_sig_power).sqrt()
            noise_sig *= gain
            mix_sig += noise_sig

        in_sig = torchaudio.functional.resample(
            mix_sig, sample_rate, hparams["sample_rate"],
        )
        yield in_sig

        out_sig = torchaudio.functional.resample(
            clean_sigs, sample_rate, hparams["sample_rate"],
        )
        # Flatten as SpeechBrain's dataloader does not support multichannel audio
        out_sig = out_sig.flatten()  # [S * T]
        yield out_sig

    sb.dataio.dataset.add_dynamic_item(
        [train_data, valid_data, test_data], audio_pipeline, takes, provides
    )

    # 3. Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from librimix_prepare import prepare_librimix as prepare_data  # noqa

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "num_speakers": hparams["num_speakers"],
            "add_noise": hparams["add_noise"],
            "version": hparams["version"],
        },
    )

    # Create the datasets objects and tokenization
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Trainer initialization
    brain = Separation(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Dynamic batching
    hparams["train_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["train_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            train_data,
            hparams["train_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=hparams["sorting"],
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["train_dataloader_kwargs"]["batch_size"] = hparams[
            "train_batch_size"
        ]
        hparams["train_dataloader_kwargs"]["shuffle"] = (
            hparams["sorting"] == "random"
        )

    hparams["valid_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["valid_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            valid_data,
            hparams["valid_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["valid_dataloader_kwargs"]["batch_size"] = hparams[
            "valid_batch_size"
        ]

    hparams["test_dataloader_kwargs"] = {
        "num_workers": hparams["dataloader_workers"]
    }
    if hparams["dynamic_batching"]:
        hparams["test_dataloader_kwargs"][
            "batch_sampler"
        ] = DynamicBatchSampler(
            test_data,
            hparams["test_max_batch_length"],
            num_buckets=hparams["num_buckets"],
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering="descending",
            max_batch_ex=hparams["max_batch_size"],
        )
    else:
        hparams["test_dataloader_kwargs"]["batch_size"] = hparams[
            "test_batch_size"
        ]

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.hparams.ter_file = os.path.join(hparams["output_folder"], "ter.txt")
    brain.evaluate(
        test_data,
        min_key="TER",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
