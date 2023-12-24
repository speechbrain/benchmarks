#!/usr/bin/env/python

"""Recipe for training an encoder-only transformer-based speech enhancement system using
discrete audio representations (see https://arxiv.org/abs/2312.09747).

The model is trained via cross-entropy loss applied to each timestep using EnCodec audio
representations (see https://arxiv.org/abs/2210.13438).

To run this recipe:
> python train_encodec.py hparams/train_encodec.yaml

Authors
 * Luca Della Libera 2023
"""

import csv
import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import if_main_process, run_on_main


class Enhancement(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass."""
        current_epoch = self.hparams.epoch_counter.current

        batch = batch.to(self.device)
        in_tokens, in_tokens_lens = batch.in_tokens

        if "embedding" in self.modules:
            # Forward encoder embedding layer
            in_embs = self.modules.embedding(in_tokens)
        else:
            # If there is no encoder embedding layer, use the codec's embeddings
            in_embs, in_embs_lens = batch.in_embs
            assert (in_tokens_lens == in_embs_lens).all()

        # Forward encoder
        enc_out = self.modules.transformer.encode(in_embs, in_tokens_lens)

        # Compute cross-entropy logits
        ce_logits = self.modules.ce_head(enc_out)

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
            hyps = ce_logits.argmax(dim=-1).tolist()
            # Remove padding (output length is equal to input length)
            hyps = [
                hyp[: len(in_token)] for hyp, in_token in zip(hyps, in_tokens)
            ]

        return ce_logits, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the objectives."""
        ce_logits, hyps = predictions

        IDs = batch.id
        _, in_tokens_lens = batch.in_tokens
        out_tokens, out_tokens_lens = batch.out_tokens
        out_tokens_list = batch.out_tokens_list

        # CE loss
        assert (in_tokens_lens == out_tokens_lens).all()
        loss = self.hparams.ce_loss(
            ce_logits.log_softmax(dim=-1), out_tokens, length=out_tokens_lens
        )

        if hyps is not None:
            assert len(hyps) == len(out_tokens_list)

            # Compute TER
            for ID, hyp, target in zip(IDs, hyps, out_tokens_list):
                self.ter_metric.append([ID], [hyp], [target])

        return loss

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

            if not if_main_process():
                return

            # Save TER
            with open(self.hparams.ter_file, "w") as w:
                self.ter_metric.write_stats(w)

            # Vocode
            if self.hparams.vocode:
                from dnsmos import DNSMOS
                from dwer import DWER

                IDs = []
                dnsmoses = []
                ref_dnsmoses = []
                dwers = []
                for score in self.ter_metric.scores:
                    ID, hyp_tokens, ref_tokens = (
                        score["key"],
                        score["hyp_tokens"],
                        score["ref_tokens"],
                    )
                    num_codebooks = 2 ** (
                        self.hparams.codec.config.target_bandwidths.index(
                            self.hparams.bandwidth
                        )
                        + 1
                    )
                    hyp_tokens = torch.as_tensor(hyp_tokens).reshape(
                        -1, num_codebooks
                    )
                    ref_tokens = torch.as_tensor(ref_tokens).reshape(
                        -1, num_codebooks
                    )

                    # Undo shift by `token_shift` to account for special tokens
                    hyp_tokens -= self.hparams.token_shift
                    assert (hyp_tokens >= 0).all()
                    ref_tokens -= self.hparams.token_shift
                    assert (ref_tokens >= 0).all()

                    # Decode
                    hyp_sig = self.hparams.codec.decode(
                        hyp_tokens[None], torch.as_tensor([1.0])
                    )[0, 0]
                    ref_sig = self.hparams.codec.decode(
                        ref_tokens[None], torch.as_tensor([1.0])
                    )[0, 0]

                    if self.hparams.save_audios:
                        save_folder = os.path.join(
                            self.hparams.output_folder, "audios"
                        )
                        os.makedirs(save_folder, exist_ok=True)
                        torchaudio.save(
                            os.path.join(save_folder, f"{ID}_hyp.wav"),
                            hyp_sig[None],
                            self.hparams.sample_rate,
                        )
                        torchaudio.save(
                            os.path.join(save_folder, f"{ID}_ref.wav"),
                            ref_sig[None],
                            self.hparams.sample_rate,
                        )

                    # Compute metrics
                    dnsmos = DNSMOS(hyp_sig, self.hparams.sample_rate)
                    ref_dnsmos = DNSMOS(ref_sig, self.hparams.sample_rate)
                    dwer = DWER(hyp_sig, ref_sig, self.hparams.sample_rate)

                    IDs.append(ID)
                    dnsmoses.append(dnsmos)
                    ref_dnsmoses.append(ref_dnsmos)
                    dwers.append(dwer)

                headers = ["ID", "DNSMOS", "RefDNSMOS", "DWER"]
                with open(
                    os.path.join(self.hparams.output_folder, "metrics.csv"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    csv_writer = csv.DictWriter(f, fieldnames=headers)
                    csv_writer.writeheader()

                    for ID, dnsmos, ref_dnsmos, dwer in zip(
                        IDs, dnsmoses, ref_dnsmoses, dwers
                    ):
                        entry = dict(
                            zip(headers, [ID, dnsmos, ref_dnsmos, dwer])
                        )
                        csv_writer.writerow(entry)

                    entry = dict(
                        zip(
                            headers,
                            [
                                "Average",
                                sum(dnsmoses) / len(dnsmoses),
                                sum(ref_dnsmoses) / len(ref_dnsmoses),
                                sum(dwers) / len(dwers),
                            ],
                        )
                    )
                    csv_writer.writerow(entry)


def dataio_prepare(hparams, codec):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"DATA_ROOT": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_max_value={"duration": hparams["train_remove_if_longer"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not run_opts.get("debug", False),
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not run_opts.get("debug", False),
        key_max_value={"duration": hparams["test_remove_if_longer"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline
    takes = ["noisy_wav", "clean_wav"]
    provides = [
        "in_tokens",
        "in_embs",
        "out_tokens_list",
        "out_tokens",
    ]

    def audio_pipeline(noisy_wav, clean_wav, is_training=True):
        # Noisy signal
        noisy_sig, sample_rate = torchaudio.load(noisy_wav)
        noisy_sig = noisy_sig[None]  # [B=1, C=1, T]
        noisy_sig = torchaudio.functional.resample(
            noisy_sig,
            sample_rate,
            hparams["sample_rate"],
        )

        # Augment if specified
        if is_training and hparams["augment"] and "augmentation" in hparams:
            noisy_sig = hparams["augmentation"](
                noisy_sig.movedim(-1, -2),
                torch.as_tensor([1.0]),
            )[0].reshape_as(noisy_sig)

        in_tokens, in_embs = codec.encode(
            noisy_sig[:, 0], torch.as_tensor([1.0])
        )
        in_tokens = in_tokens.flatten()
        in_tokens += hparams[
            "token_shift"
        ]  # Shift by `token_shift` to account for special tokens
        yield in_tokens

        in_embs = in_embs[0]
        yield in_embs

        # Clean signal
        clean_sig, sample_rate = torchaudio.load(clean_wav)
        clean_sig = clean_sig[None]  # [B=1, C=1, T]
        clean_sig = torchaudio.functional.resample(
            clean_sig,
            sample_rate,
            hparams["sample_rate"],
        )
        out_tokens, _ = codec.encode(clean_sig[:, 0], torch.as_tensor([1.0]))
        out_tokens = out_tokens.flatten()
        out_tokens += hparams[
            "token_shift"
        ]  # Shift by `token_shift` to account for special tokens

        out_tokens_list = out_tokens.tolist()
        yield out_tokens_list

        out_tokens = torch.LongTensor(out_tokens_list)
        yield out_tokens

    sb.dataio.dataset.add_dynamic_item(
        [train_data], audio_pipeline, takes, provides
    )
    sb.dataio.dataset.add_dynamic_item(
        [valid_data, test_data],
        lambda *args, **kwargs: audio_pipeline(
            *args, **kwargs, is_training=False
        ),
        takes,
        provides,
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
    from voicebank_prepare import prepare_voicebank as prepare_data  # noqa

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "num_valid_speakers": hparams["num_valid_speakers"],
        },
    )

    # Define codec
    codec = hparams["codec"]

    # Create the datasets objects and tokenization
    train_data, valid_data, test_data = dataio_prepare(hparams, codec)

    # Pretrain the specified modules
    run_on_main(hparams["pretrainer"].collect_files)
    run_on_main(hparams["pretrainer"].load_collected)

    # Trainer initialization
    brain = Enhancement(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Add objects to trainer
    brain.codec = codec

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
