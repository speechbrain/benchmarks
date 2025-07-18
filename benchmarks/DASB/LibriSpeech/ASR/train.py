#!/usr/bin/env/python3
"""Recipe for training an discrete tokens ctc ASR system with librispeech.

Decoding is performed with greedy decoding at validation time.
At test time, beamsearch is used with an optional external language model.

Authors
 * Pooneh Mousavi 2024
 * Jarod Duret 2024
"""

import os
import sys
import time
import torch
import torchaudio
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from speechbrain.tokenizers.SentencePiece import SentencePiece
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)


logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        in_toks, _ = batch.speech_tokens

        in_embs = self.modules.discrete_embedding_layer(
            in_toks
        )  # [B, T, N-Q, D]

        # Attention-Pooling
        att_w = self.modules.attention_mlp(in_embs)  # [B, T, N-Q, 1]
        in_embs = torch.matmul(att_w.transpose(2, -1), in_embs).squeeze(
            -2
        )  # [B, T, D]

        # forward modules
        if type(self.modules.encoder).__name__ == "ContextNet":
            enc_out = self.modules.encoder(in_embs)

        elif type(self.modules.encoder).__name__ == "LSTM":
            enc_out, _ = self.modules.encoder(in_embs)

        else:
            raise NotImplementedError

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        p_tokens = None
        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions
        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )
        elif stage == sb.Stage.TEST:
            predicted_words = [
                hyp[0].text.split(" ") for hyp in predicted_tokens
            ]

        if stage != sb.Stage.TRAIN:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            if type(self.hparams.scheduler).__name__ == "NewBobScheduler":
                lr, new_lr = self.hparams.scheduler(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif type(self.hparams.scheduler).__name__ == "LinearNoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                raise NotImplementedError

            optimizer = self.optimizer.__class__.__name__
            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.output_wer_folder, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if (
            should_step
            and type(self.hparams.scheduler).__name__ == "LinearNoamScheduler"
        ):
            self.hparams.scheduler(self.optimizer)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 1. Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("speech_tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens

    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        info = torchaudio.info(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        # resampled = resampled.unsqueeze(0)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens", "speech_tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_val = hparams["dynamic_batch_sampler_val"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_val,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["cached_data_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["cached_data_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Use pretrained embeddings
    if hparams["pretrain_embeddings"]:
        tokens_loader = hparams["tokens_loader"]
        embs = tokens_loader.load_pretrained_embeddings(
            hparams["pretain_embeddings_folder"]
        )
        if isinstance(hparams["num_codebooks"], int):
            embs = embs[
                : hparams["num_codebooks"] * hparams["vocab_size"],
            ]
        # For discrete SSL, num_codebooks is a list used to determine which layers to use.
        # It is not sequential and can be, for example, [0, 1] or [1, 4].
        elif isinstance(hparams["num_codebooks"], list):
            indices = [
                i
                for codebook_idx in hparams["num_codebooks"]
                for i in range(
                    codebook_idx * hparams["vocab_size"],
                    (codebook_idx + 1) * hparams["vocab_size"],
                )
            ]
            indices = torch.tensor(indices, dtype=torch.long)
            embs = embs[indices]
        hparams["discrete_embedding_layer"].init_embedding(embs)

    # Log number of parameters/buffers
    model_params = sum(
        [
            x.numel()
            for module in hparams["modules"].values()
            for x in module.state_dict().values()
        ]
    )
    hparams["train_logger"].log_stats(
        stats_meta={
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer
    vocab_list = [
        tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())
    ]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"], vocab_list=vocab_list,
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Measure time
    start_time = time.time()  # Start the timer
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    end_time = time.time()  # End the timer
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    logger.info(f"Model execution time: {elapsed_time:.6f} seconds")

    if hparams["testing"]:
        # Testing
        if not os.path.exists(hparams["output_wer_folder"]):
            os.makedirs(hparams["output_wer_folder"])

        for k in test_datasets.keys():  # keys are test_clean, test_other etc
            asr_brain.hparams.output_wer_folder = os.path.join(
                hparams["output_wer_folder"], f"wer_{k}.txt"
            )
            asr_brain.evaluate(
                test_datasets[k],
                test_loader_kwargs=hparams["test_dataloader_opts"],
                min_key="WER",
            )
