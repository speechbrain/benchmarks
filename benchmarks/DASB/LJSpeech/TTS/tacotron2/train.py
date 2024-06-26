# -*- coding: utf-8 -*-
"""
 Recipe for training the Tacotron Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system

 To run this recipe, do the following:
 # python train.py --device=cuda:0 --max_grad_norm=1.0 --data_folder=/your_folder/LJSpeech-1.1 hparams/train.yaml

 to infer simply load saved model and do
 savemodel.infer(text_Sequence,len(textsequence))

 were text_Sequence is the output of the text_to_sequence function from
 textToSequence.py (from textToSequence import text_to_sequence)

 Authors
 * Georges Abous-Rjeili 2021
 * Artem Ploujnikov 2021
 * Yingzhi Wang 2022
"""
import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from benchmarks.DASB.utils.preparation import add_prepared_features
from benchmarks.DASB.utils.hparams import as_list
from speechbrain.utils.data_utils import scalarize
from pathlib import Path

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_COUNT = 1


class Tacotron2Brain(sb.Brain):
    """The Brain implementation for Tacotron2"""

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        if self.hparams.squish_layers:
            self.layer_weights = self._get_squish_layer_weights()
        self.layer_idx = self._get_selected_layer_idx()
        return super().on_fit_start()

    def _get_squish_layer_weights(self):
        layer_weights = self.hparams.squish_layers_weights
        if isinstance(layer_weights, str):
            layer_weights = [
                float(weight) for weight in layer_weights.split(",")
            ]
        layer_weights = torch.tensor(layer_weights)[None, None, :, None].to(
            self.device
        )
        layer_weights = layer_weights / layer_weights.sum()
        return layer_weights

    def _get_selected_layer_idx(self):
        selected_layers = None
        if self.hparams.select_layers:
            layers = as_list(self.hparams.select_layers, dtype=int)
            model_layers_map = {
                layer: idx
                for idx, layer in enumerate(
                    as_list(self.hparams.ssl_model_layers)
                )
            }
            selected_layers = [model_layers_map[layer] for layer in layers]
        return selected_layers

    def select_layers(self, audio_ssl):
        """Applies layer squishing, if enabled

        Arguments
        ---------
        audio_ssl : torch.Tensor
            SSL features

        Returns
        -------
        audio_ssl : torch.Tensor
            SSL features, squished if enabled
        """
        if self.hparams.select_layers:
            audio_ssl = audio_ssl[:, :, self.layer_idx, :]
        if self.hparams.squish_layers:
            audio_ssl = (audio_ssl * self.layer_weights).sum(
                dim=2, keepdim=True
            )
        return audio_ssl

    def compute_forward(self, batch, stage):
        """Computes the forward pass

        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage

        Returns
        -------
        the model output
        """
        batch = batch.to(self.device)
        audio_ssl = self.select_layers(batch.audio_ssl.data)
        batch_size, audio_max_len, audio_heads, audio_feat = audio_ssl.shape
        tokens_max_len = batch.tokens.data.size(1)
        inputs = (
            batch.tokens.data,
            batch.tokens.lengths * tokens_max_len,
            audio_ssl.view(
                batch_size, audio_max_len, audio_heads * audio_feat
            ).transpose(-1, -2),
            batch.audio_ssl.data.size(1),
            batch.audio_ssl.lengths * audio_max_len,
        )

        max_input_length = batch.tokens.data.size(1)
        return self.modules.model(inputs, alignments_dim=max_input_length)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.lr_annealing(self.optimizer)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        loss = self._compute_loss(predictions, batch, stage)
        return loss

    def _compute_loss(self, predictions, batch, stage):
        """Computes the value of the loss function and updates stats

        Arguments
        ---------
        predictions: tuple
            model predictions
        batch: PaddedBatch
            Inputs for this training iteration.
        stage: sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss: torch.Tensor
            the loss value
        """
        tokens_len = batch.tokens.data.size(1)
        audio = self.select_layers(batch.audio_ssl.data)
        batch_size, audio_max_len, audio_heads, audio_feat = audio.shape
        audio = audio.view(
            batch_size, audio_max_len, audio_heads * audio_feat
        ).transpose(-1, -2)
        audio_len = batch.audio_ssl.lengths * audio_max_len
        gate_targets = (
            torch.arange(audio_max_len, device=audio.device)[None, :]
            >= audio_len[:, None] - 1
        ).float()
        loss_stats = self.hparams.criterion(
            predictions,
            (audio, gate_targets),
            batch.tokens.lengths * tokens_len,
            audio_len,
            self.last_epoch,
        )
        self.last_loss_stats[stage] = scalarize(loss_stats)
        return loss_stats.loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        self.modules.vocoder.device = self.device
        self.create_perfect_samples()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            lr = self.optimizer.param_groups[-1]["lr"]
            self.last_epoch = epoch

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )

            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            self.checkpointer.save_and_keep_only(
                meta=epoch_metadata,
                min_keys=["loss"],
                ckpt_predicate=(
                    (
                        lambda ckpt: (
                            ckpt.meta["epoch"]
                            % self.hparams.keep_checkpoint_interval
                            != 0
                        )
                    )
                    if self.hparams.keep_checkpoint_interval is not None
                    else None
                ),
            )
            # Create audio samples
            self.create_samples()

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )

    def create_samples(self):
        """Writes audio samples at the end of an epoch"""
        epoch = self.hparams.epoch_counter.current
        if epoch % self.hparams.samples_interval != 0:
            return
        sample_loader = sb.dataio.dataloader.make_dataloader(
            self.sample_data, **self.hparams.sample_dataloader_opts,
        )
        with self.hparams.progress_report:
            for batch in sample_loader:
                batch = batch.to(self.device)
                tokens, tokens_length = batch.tokens
                tokens_max_len = tokens.size(1)
                audio_ssl, audio_lengths, alignments = self.modules.model.infer(
                    inputs=tokens, input_lengths=tokens_length * tokens_max_len
                )
                audio_ssl = audio_ssl.transpose(-1, -2)
                batch_size, max_len, feat_dim = audio_ssl.shape
                audio_ssl = audio_ssl.view(
                    batch_size,
                    max_len,
                    self.hparams.audio_tokens_per_step,
                    self.hparams.audio_dim,
                )
                wav = self.modules.vocoder(audio_ssl).squeeze(1)
                self.hparams.progress_report.write(
                    ids=batch.uttid,
                    audio=wav,
                    length_pred=audio_lengths,
                    length=batch.audio_ssl.lengths,
                    tgt_max_length=batch.audio_ssl.data.size(1),
                    alignments=alignments,
                )

    def create_perfect_samples(self):
        """Creates the best samples that can be created using
        the vocoder provided, for comparison purposes"""
        if not self.hparams.progress_logger["perfect_samples_created"]:
            sample_loader = sb.dataio.dataloader.make_dataloader(
                self.sample_data, **self.hparams.sample_dataloader_opts
            )
            for batch in sample_loader:
                batch = batch.to(self.device)
                sample_ssl, length = batch.audio_ssl
                if self.hparams.vocoder_flat_feats:
                    sample_ssl = sample_ssl.flatten(start_dim=-2)
                samples = self.modules.vocoder(sample_ssl)
                sample_ssl = self.select_layers(sample_ssl)
                samples = samples.squeeze(1)
                max_len = samples.size(1)
                samples_length_abs = (batch.audio_ssl.lengths * max_len).int()
                with self.hparams.progress_logger:
                    for item_id, item_wav, item_length in zip(
                        batch.uttid, samples, samples_length_abs
                    ):
                        item_cut = item_wav[: item_length.item()]
                        self.hparams.progress_logger.save(
                            name=f"{item_id}.wav",
                            content=item_cut.detach().cpu(),
                            mode="audio",
                            folder="_perfect",
                            samplerate=self.hparams.model_sample_rate,
                        )
                    self.hparams.progress_logger[
                        "perfect_samples_created"
                    ] = True
                    self.hparams.progress_logger.clear()


def dataio_prepare(hparams):
    # Define audio pipeline:
    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }

    label_encoder = hparams["label_encoder"]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_norm")
    def label_norm_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label.upper()

    @sb.utils.data_pipeline.takes("label_norm")
    @sb.utils.data_pipeline.provides("tokens", "tokens_len")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        tokens = label_encoder.encode_sequence_torch(label)
        yield tokens
        yield len(tokens)

    init_sequence_encoder(hparams)

    for dataset in hparams["splits"]:
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[label_norm_pipeline, tokens_pipeline],
            replacements={"data_root": hparams["data_folder"]},
            output_keys=["uttid", "audio_ssl", "tokens"],
        )

        add_prepared_features(
            dataset=dynamic_dataset,
            save_path=Path(hparams["prepare_save_folder"]) / "features",
            id_key="uttid",
            features=["audio_ssl"],
        )
        dynamic_dataset = dynamic_dataset.filtered_sorted(
            sort_key="tokens_len", reverse=True
        )
        datasets[dataset] = dynamic_dataset

    datasets["sample"] = (
        datasets["valid"]
        .batch_shuffle(1)
        .filtered_sorted(
            select_n=hparams["num_audio_samples"],
            sort_key="tokens_len",
            reverse=True,
        )
    )

    return datasets


def init_sequence_encoder(hparams):
    """Initialize a sequence encoder

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    prefix: str
        the prefix to be prepended to hyperparameter keys, per the naming
        convention

        {prefix}_label_encoder: the hparams key for the label encoder
        {prefix}_list_file:  the hparams key for the list file

    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder instance"""
    encoder = hparams["label_encoder"]
    token_list_file_name = hparams["token_list_file"]
    tokens = read_token_list(token_list_file_name)
    encoder.add_unk()
    encoder.update_from_iterable(tokens, sequence_input=False)
    encoder.expect_len(len(tokens) + SPECIAL_TOKEN_COUNT)
    return encoder


def read_token_list(file_name):
    """Reads a simple text file with tokens (e.g. characters or phonemes) listed
    one per line

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    result: list
        a list of tokens
    """
    if not Path(file_name).exists():
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip("\r\n") for line in token_file if line]


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech

    if not hparams["skip_prep"]:
        with hparams["freezer"]:
            sb.utils.distributed.run_on_main(
                prepare_ljspeech,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_folder": hparams["prepare_save_folder"],
                    "splits": hparams["splits"],
                    "split_ratio": hparams["split_ratio"],
                    "seed": hparams["seed"],
                    "skip_prep": hparams["skip_prep"],
                    "extract_features": ["audio_ssl", "audio_ssl_len"],
                    "extract_features_opts": hparams["extract_features_opts"],
                    "frozen_split_path": hparams.get("frozen_split_path"),
                    "model_name": "Tacotron2SSL",
                    "skip_ignore_folders": hparams[
                        "prepare_skip_ignore_folders"
                    ],
                    "device": run_opts.get("device", "cpu"),
                },
            )

    datasets = dataio_prepare(hparams)

    # Brain class initialization
    tacotron2_brain = Tacotron2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    tacotron2_brain.sample_data = datasets["sample"]

    # Training
    tacotron2_brain.fit(
        tacotron2_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        tacotron2_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
