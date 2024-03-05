#!/usr/bin/env/python

"""Recipe for training a speech enhancement system using semantic audio tokens.

The model is trained via cross-entropy loss applied to each timestep.

The neural network architecture is inspired by:
https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py#L27

To run this recipe:
> python train_discrete.py hparams/<config>.yaml

Authors
 * Luca Della Libera 2024
"""

import csv
import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import write_audio
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import if_main_process, run_on_main


class Enhancement(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        current_epoch = self.hparams.epoch_counter.current

        batch = batch.to(self.device)
        in_sig, in_sig_lens = batch.in_sig
        out_sig, out_sig_lens = batch.out_sig

        # Augment if specified
        if stage == sb.Stage.TRAIN and self.hparams.augment:
            in_sig, in_sig_lens = self.hparams.augmentation(in_sig, in_sig_lens)

        # Extract audio tokens
        assert (in_sig_lens == out_sig_lens).all()
        sig, lens = (
            torch.cat([in_sig, out_sig]),
            torch.cat([in_sig_lens, out_sig_lens]),
        )
        with torch.no_grad():
            self.hparams.codec.to(self.device).eval()
            tokens, _, _ = self.hparams.codec(
                sig,
                lens,
                self.hparams.layer_ids,
                deduplicates=[False] * len(self.hparams.layer_ids),
                bpe_tokenizers=[None] * len(self.hparams.layer_ids),
            )
        in_tokens = tokens[: len(tokens) // 2]
        batch.out_tokens = tokens[len(tokens) // 2 :], out_sig_lens

        # Forward embedding layer (one for each codebook)
        in_tokens += torch.arange(
            0,
            self.hparams.num_codebooks * self.hparams.vocab_size,
            self.hparams.vocab_size,
            device=self.device,
        )  # Offset to select embeddings from the correct codebook
        in_embs = self.modules.embedding(in_tokens).reshape(
            len(batch),
            -1,
            self.hparams.embedding_dim,
            self.hparams.num_codebooks,
        )

        # Reduce to 1 channel
        in_embs = in_embs.sum(dim=-1)

        # Forward encoder
        if type(self.modules.encoder).__name__ == "CRDNN":
            enc_out = self.modules.encoder(in_embs)
        elif type(self.modules.encoder).__name__ == "TransformerASR":
            enc_out = self.modules.encoder.encode(in_embs, in_sig_lens)
        else:
            raise NotImplementedError

        # Compute cross-entropy logits (one for each codebook)
        ce_logits = self.modules.ce_head(enc_out).reshape(
            len(batch), -1, self.hparams.num_codebooks, self.hparams.vocab_size,
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
            min_length = self.hparams.num_codebooks
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

        # Cross-entropy loss
        loss = self.hparams.ce_loss(
            ce_logits.log_softmax(dim=-1).flatten(start_dim=-3, end_dim=-2),
            out_tokens.flatten(start_dim=-2),
            length=out_tokens_lens,
        )

        if hyps is not None:
            targets = out_tokens.flatten(start_dim=-2).tolist()

            # Remove padding
            min_length = self.hparams.num_codebooks
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

        from dnsmos import DNSMOS
        from dwer import DWER
        from spk_sim import SpkSimECAPATDNN, SpkSimWavLM

        metrics = []
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
            hyp_tokens = torch.as_tensor(
                hyp_tokens, device=self.device
            ).reshape(-1, self.hparams.num_codebooks)
            rec_tokens = torch.as_tensor(
                rec_tokens, device=self.device
            ).reshape(-1, self.hparams.num_codebooks)

            with torch.no_grad():
                self.hparams.vocoder.device = self.device
                self.hparams.vocoder.to(self.device).eval()
                self.hparams.vocoder.tokenize = False
                offsets = (
                    torch.arange(
                        len(self.hparams.layer_ids), device=self.device
                    )
                    * self.hparams.vocab_size
                )
                hyp_sig = self.hparams.vocoder.decode_batch(
                    hyp_tokens[None] + offsets + 1
                )[0, 0]
                rec_sig = self.hparams.vocoder.decode_batch(
                    rec_tokens[None] + offsets + 1
                )[0, 0]

            ref_sig = self.test_set[self.test_set.data_ids.index(ID)][
                "out_sig"
            ].to(
                self.device
            )  # Original output signal (resampled)
            in_sig = self.test_set[self.test_set.data_ids.index(ID)][
                "in_sig"
            ]  # Original input signal (resampled)

            if self.hparams.save_audios:
                save_folder = os.path.join(self.hparams.output_folder, "audios")
                os.makedirs(save_folder, exist_ok=True)
                write_audio(
                    os.path.join(save_folder, f"{ID}_hyp.wav"),
                    hyp_sig.cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{ID}_rec.wav"),
                    rec_sig.cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{ID}_ref.wav"),
                    ref_sig.cpu(),
                    self.hparams.sample_rate,
                )
                write_audio(
                    os.path.join(save_folder, f"{ID}_in.wav"),
                    in_sig.cpu(),
                    self.hparams.sample_rate,
                )

            # Compute metrics
            min_length = min(len(hyp_sig), len(ref_sig))
            sisnr = -si_snr_loss(
                hyp_sig[None, :min_length],
                ref_sig[None, :min_length],
                torch.as_tensor([1.0], device=self.device),
            ).item()
            spk_sim_ecapatdnn = SpkSimECAPATDNN(
                hyp_sig, ref_sig, self.hparams.sample_rate
            )
            spk_sim_wavlm = SpkSimWavLM(
                hyp_sig, ref_sig, self.hparams.sample_rate
            )
            dnsmos = DNSMOS(hyp_sig, self.hparams.sample_rate)
            rec_dnsmos = DNSMOS(rec_sig, self.hparams.sample_rate)
            ref_dnsmos = DNSMOS(ref_sig, self.hparams.sample_rate)
            dwer, avg_dwer, text, ref_text = DWER(
                hyp_sig, ref_sig, self.hparams.sample_rate
            )
            metrics.append(
                [
                    ID,
                    sisnr,
                    spk_sim_ecapatdnn,
                    spk_sim_wavlm,
                    *dnsmos,
                    *rec_dnsmos,
                    *ref_dnsmos,
                    dwer,
                    avg_dwer,
                    text,
                    ref_text,
                ]
            )

        headers = ["ID", "SI-SNR", "SpkSimECAPATDNN", "SpkSimWavLM"]
        headers += ["SigMOS", "BakMOS", "OvrMOS", "p808MOS"]
        headers += ["RecSigMOS", "RecBakMOS", "RecOvrMOS", "Recp808MOS"]
        headers += ["RefSigMOS", "RefBakMOS", "RefOvrMOS", "Refp808MOS"]
        headers += ["dWER", "dWERAvg", "text", "ref_text"]
        with open(
            os.path.join(self.hparams.output_folder, "metrics.csv"),
            "w",
            encoding="utf-8",
        ) as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()

            for entry in metrics:
                entry = dict(zip(headers, entry))
                csv_writer.writerow(entry)

            columns = list(zip(*metrics))
            entry = dict(
                zip(
                    headers,
                    ["Average"]
                    + [sum(c) / len(c) for c in columns[1:-3]]
                    + [avg_dwer]
                    + ["", ""],
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

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
        replacements={"data_root": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_max_value={"duration": hparams["train_remove_if_longer"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
        replacements={"data_root": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"data_root": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["test_remove_if_longer"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline
    takes = ["wav", "noise", "snr"]
    provides = ["in_sig", "out_sig"]

    def audio_pipeline(wav, noise, snr):
        # Clean signal
        clean_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        clean_sig = sb.dataio.dataio.read_audio(wav)
        clean_sig = torchaudio.functional.resample(
            clean_sig, clean_sample_rate, hparams["sample_rate"],
        )

        # Noise signal
        noise_sample_rate = sb.dataio.dataio.read_audio_info(noise).sample_rate
        noise_sig = sb.dataio.dataio.read_audio(noise)
        noise_sig = torchaudio.functional.resample(
            noise_sig, noise_sample_rate, hparams["sample_rate"],
        )

        # Adjust length if needed
        if len(noise_sig) > len(clean_sig):
            noise_sig = noise_sig[: len(clean_sig)]
        elif len(noise_sig) < len(clean_sig):
            num_reps = len(clean_sig) // len(noise_sig) + 1
            noise_sig = noise_sig.repeat(num_reps)
            noise_sig = noise_sig[: len(clean_sig)]

        # Noisy signal (mixing)
        noise_gain = -float(snr)
        clean_sig_power = (clean_sig ** 2).mean()
        ratio = 10 ** (
            noise_gain / 10
        )  # ratio = noise_sig_power / clean_sig_power
        desired_noise_sig_power = ratio * clean_sig_power
        noise_sig_power = (noise_sig ** 2).mean()
        gain = (desired_noise_sig_power / (noise_sig_power + 1e-8)).sqrt()
        noise_sig *= gain
        noisy_sig = clean_sig + noise_sig

        in_sig = noisy_sig
        yield in_sig

        out_sig = clean_sig
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

    # Dataset preparation
    from ljspeechwham_prepare import (
        prepare_ljspeechwham as prepare_data,
    )  # noqa

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
        },
    )

    # Create the datasets objects and tokenization
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Trainer initialization
    brain = Enhancement(
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
