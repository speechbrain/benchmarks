#!/usr/bin/env/python3
"""Recipe for training a Text-to-Speech system based on tokenized audio

Inspired by WhisperSpeech
https://github.com/collabora/WhisperSpeech

However, this is not an implementation of WhisperSpeech, but rather
a radical simplification of it that uses only an acoustic model


Authors
 * Artem Ploujnikov 2024
"""


import logging
import speechbrain as sb
import math
import torch
import sys
from functools import partial
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.dataio.dataio import clean_padding_
from speechbrain.utils.distributed import run_on_main
import re
import string

base_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(base_dir)

from model.Tokotron import (
    RepresentationMode,
    get_silence_repr,
    get_silence_token,
    use_silence_padding,
    feature_pad_to,
)  # noqa: E402
from evaluate import TokotronEvaluator  # noqa: E402

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_COUNT = 1


# Brain class for speech recognition training
class TokotronBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super().__init__(
            modules, opt_class, hparams, run_opts, checkpointer,
        )
        self.evaluator = TokotronEvaluator(
            hparams=hparams,
            create_waveform_fn=self.create_waveform,
            device=self.device,
        )
        self.representation_mode = RepresentationMode(
            self.hparams.representation_mode
        )

    def create_waveform(self, audio, length, emb):
        """Creates a waveform from a discrete or continuous audio
        representation

        Arguments
        ---------
        audio : torch.Tensor
            An audio tensor (Batch x Length x Heads or Batch x Length x Heads x Features)
        lengths : torch.Tensor
            A 1-D tensor
        emb: dict
            Embeddings (speaker, etc)

        Returns
        -------
        wav : torch.Tensor
        """
        self.modules.tokenizer.device = self.device
        if hasattr(self.modules.tokenizer, "codec_vocoder"):
            self.modules.tokenizer.codec_vocoder.to(self.device)
            self.modules.tokenizer.codec_vocoder.device = self.device
        with torch.no_grad():
            wav = self.modules.tokenizer.tokens_to_sig(
                audio, **self.token_model_kwargs
            )
            clean_padding_(wav, length)
            wav = wav.to(self.device)
        return wav

    def compute_forward(self, batch, stage):
        """Runs all the computation of the Tokotron TTS

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            TTS predictions
        """
        batch = batch.to(self.device)
        tokens, tokens_length = batch.tokens
        features = self.prepare_features(batch)
        (
            audio_bos,
            audio_bos_length,
            audio_tgt,
            audio_tgt_length,
            spk_emb,
        ) = features

        predictions = self.modules.model(
            input_tokens=tokens,
            input_length=tokens_length,
            audio=audio_bos,
            audio_length=audio_bos_length,
            emb={"spk": spk_emb},
        )

        return predictions, features

    def prepare_features(self, batch):
        if self.hparams.spk_emb_shuffle:
            wav, wav_length = batch.spk_emb_random_match
        else:
            wav, wav_length = batch.sig
        spk_emb = self._compute_spk(wav, wav_length).squeeze(1)

        if self.representation_mode == RepresentationMode.DISCRETE:
            audio_bos, audio_bos_length = batch.audio_bos
            audio_tgt, audio_tgt_length = batch.audio_pad
        else:
            wav, audio_length = batch.sig
            audio = self.modules.ssl_model(wav)
            audio = audio[self.hparams.ssl_model_layers, :, :, :].permute(
                1, 2, 0, 3
            )
            batch_size, _, heads, dim = audio.shape
            bos = torch.zeros_like(audio[:, :1, :, :]).reshape(
                batch_size, self.hparams.bos_width, heads, dim
            )
            audio_bos = torch.concatenate([bos, audio], dim=1)
            audio_bos_length = audio_length * audio.size(1) / audio_bos.size(1)
            audio_tgt = audio
            audio_tgt_length = audio_length

        return audio_bos, audio_bos_length, audio_tgt, audio_tgt_length, spk_emb

    def _compute_spk(self, wav, wav_length):
        mel_spec = self.spk_emb_model.mel_spectogram(wav.squeeze(1))
        spk_emb_pred = self.spk_emb_model.encode_mel_spectrogram_batch(
            mel_spec, wav_length
        )
        return spk_emb_pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
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

        predictions, features = predictions
        (
            audio_bos,
            audio_bos_length,
            audio_tgt,
            audio_tgt_length,
            spk_emb,
        ) = features

        loss_details = self.hparams.compute_cost(
            predictions=predictions,
            audio=audio_tgt,
            audio_length=audio_tgt_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
        )
        self.loss_metric.append(
            batch.uttid,
            predictions=predictions,
            audio=audio_tgt,
            audio_length=audio_tgt_length,
            input_tokens=batch.tokens.data,
            input_length=batch.tokens.lengths,
            reduction="batch",
        )
        return loss_details.loss

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
        if hasattr(self.modules, "vocoder") and hasattr(
            self.modules.vocoder, "model"
        ):
            self.modules.vocoder.model.device = self.device
        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost, batch_eval=True,
        )
        if (
            self.hparams.audio_emb_pretrained
            and epoch == 1
            and stage == sb.Stage.TRAIN
        ):
            # TODO: Clean this up
            if hasattr(self.hparams.token_model, "vocabulary"):
                vocabulary = self.hparams.token_model.vocabulary
            elif hasattr(self.hparams.token_model, "vocabularies"):
                vocabulary = torch.stack(
                    [
                        torch.from_numpy(voc)
                        for voc in self.hparams.token_model.vocabularies
                    ]
                )
            self.modules.model.init_audio_emb(vocabulary)
        # Load the compression model only if compression is enables
        pretrained_run_opts = {"device": self.device}
        self.spk_emb_model = self.hparams.spk_emb_model(
            run_opts=pretrained_run_opts
        )
        self.representation_mode = RepresentationMode(
            self.hparams.representation_mode
        )
        # If speaker embedding shuffling is enabled, re-initialize them for the
        # epoch
        if self.hparams.spk_emb_shuffle:
            stage_key = stage.name.lower()
            self.resample_fn[stage_key](epoch=epoch)

        # Reset the learning rate - if supported. This is useful when fine-tuning
        # a model pre-trained on another dataset
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.reset_annealing_epoch is not None
            and epoch is not None
            and epoch == self.hparams.reset_annealing_epoch
        ):
            self.hparams.lr_annealing.n_steps = 0

        self.is_evaluating = False
        if stage == sb.Stage.VALID:
            if self.is_eval_epoch(epoch):
                self.evaluator.on_evaluate_start(stage, epoch)
                self.is_evaluating = True
            else:
                logger.info("No evaluation on epoch %d", epoch)
        elif stage == sb.Stage.TEST:
            self.evaluator.on_evaluate_start(stage, epoch)
            self.is_evaluating = True
        self.token_model_kwargs = getattr(
            self.hparams, "token_model_kwargs", {}
        )

    def is_eval_epoch(self, epoch):
        """Determines whether or not evaluation should be performed
        in the specieied epoch

        Arguments
        ---------
        epoch : int
            The epoch number. If omitted, the epoch number from the
            epoch counter will be used

        Returns
        -------
        eval_epoch : bool
            True if evaluation should be run in this epoch, false
            otherwise"""
        if epoch is None:
            epoch = self.hparams.epoch_counter.current
        return epoch % self.hparams.eval_interval == 0

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit/compiled modules
        # cannot be pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None and not getattr(
            self, "_ckpt_recovered", False
        ):
            checkpoint = self.checkpointer.recover_if_possible()
            if not checkpoint:
                self.check_init()
            self._ckpt_recovered = True

    def check_init(self):
        init_from = getattr(self.hparams, "init_from", None)
        if init_from is not None:
            logger.info("Initializing with pre-trained weights from %s", init_from)
            init_from_path = Path(init_from)
            model_path = init_from_path / "model.ckpt"
            with open(model_path, "rb") as model_file:
                model_state_dict = torch.load(model_file, map_location=self.device)
                tgt_state_dict = self.modules.model.state_dict()
                ignore_keys = []
                for k, v in model_state_dict.items():
                    if k in tgt_state_dict and tgt_state_dict[k].shape != v.shape:
                        logger.warning("Ignoring shape mismatch for %s", k)
                        ignore_keys.append(k)
                for k in ignore_keys:
                    del model_state_dict[k]
                self.modules.model.load_state_dict(model_state_dict, strict=False)
            logger.info("Successfully initialized with pre-trained weights from %s", init_from)

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        if self.is_evaluating:
            self.evaluator.evaluate_batch(batch)
        return loss.detach().cpu()

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
        loss_stats = self.loss_metric.summarize(flat=True)
        stage_stats = {"loss": stage_loss, **loss_stats}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # End evaluation and report stats
        if stage != sb.Stage.TRAIN and self.is_eval_epoch(epoch):
            self.evaluator.on_evaluate_end()
            eval_summary = self.evaluator.compute_summary()
            eval_summary_stats = {
                key: eval_summary.get(value)
                for key, value in self.hparams.eval_summary_log.items()
            }
            stage_stats.update(eval_summary_stats)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            if self.hparams.lr_annealing_mode == "epoch":
                _, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            lr = self.optimizer.param_groups[0]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

    def fit_batch(self, batch):
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss


INPUT_FEATURE_MAP = {"text": "label_norm", "phonemes": "phn"}


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    silence_token : dict
        the token used for silence
    """

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    representation_mode = RepresentationMode(
        hparams.get("representation_mode", RepresentationMode.DISCRETE)
    )

    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    label_encoder = hparams["label_encoder"]
    input_feature = INPUT_FEATURE_MAP[hparams["input"]]

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_ref_pipeline(wav):
        """The audio loading pipeline for references

        Arguments
        ---------
        wav : strƒnum_
            The file path

        Returns
        -------
        sig : torch.Tensor
            The waveform
        """
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_norm", "label_norm_eval")
    def text_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        label_norm = label.upper()
        yield label.upper()
        label_norm_eval = RE_PUNCTUATION.sub("", label_norm)
        yield label_norm_eval

    @sb.utils.data_pipeline.takes(input_feature)
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label_encoder.encode_sequence_torch(label)

    use_silence_padding = hparams.get("use_silence_padding", True)
    if "token_model_layers" in hparams:
        audio_tokens_per_step = len(hparams["token_model_layers"])
    else:
        audio_tokens_per_step = hparams["audio_tokens_per_step"]
    layer_idx = None
    if "speech_model_layers" in hparams:
        layer_idx = get_selected_layer_indexes(hparams)
    if use_silence_padding:
        if representation_mode == RepresentationMode.DISCRETE:
            silence_padding = get_silence_token(
                hparams["tokenizer"],
                num_codebooks=(
                    hparams["speech_model_layers"]
                    if "speech_model_layers" in hparams
                    else audio_tokens_per_step
                )
            )
        else:
            silence_padding = get_silence_repr(hparams["ssl_model"],)
    else:
        silence_padding = (
            torch.ones(audio_tokens_per_step, dtype=torch.int64)
            * hparams["eos_index"]
        )

    silence_padding = silence_padding.cpu()
    if layer_idx:
        silence_padding = silence_padding[layer_idx]
    else:
        silence_padding = silence_padding[:audio_tokens_per_step]
    silence_padding_len = int(math.ceil(hparams["silence_padding"]))
    bos_width = hparams.get("bos_width", 1)
    audio_bos_prefix = (
        torch.ones(bos_width, audio_tokens_per_step) * hparams["bos_index"]
    )
    if representation_mode == RepresentationMode.CONTINUOUS:
        audio_bos_prefix = audio_bos_prefix.unsqueeze(-1).repeat(
            1, 1, hparams["audio_dim"]
        )

    tokens_loader = hparams.get("tokens_loader")
    if layer_idx is not None:
        tokens_loader_kwargs = {
            "num_codebooks": layer_idx
        }
    else:
        tokens_loader_kwargs = {"num_codebooks": audio_tokens_per_step}    

    @sb.utils.data_pipeline.takes("uttid")
    @sb.utils.data_pipeline.provides("audio_pad", "audio_bos")
    def audio_pipeline(id):
        audio = tokens_loader.tokens_by_uttid(id, **tokens_loader_kwargs)
        audio_pad = feature_pad_to(
            audio, len(audio) + silence_padding_len, silence_padding
        )
        yield audio_pad
        audio_bos = torch.cat([audio_bos_prefix, audio_pad], dim=0)
        yield audio_bos

    def spk_emb_random_match(uttid, dataset, spk_sample):
        # Sample a speaker-matched embedding
        selected_idx = spk_sample[uttid]

        # Retrieve the embedding value from the dataset
        with dataset.output_keys_as(["sig"]):
            spk_emb = dataset[selected_idx]["sig"]
        return spk_emb

    dynamic_items = [
        text_pipeline,
        tokens_pipeline,
        audio_ref_pipeline,
        audio_pipeline,
    ]
    output_keys = [
        "uttid",
        "tokens",
        "audio_pad",
        "audio_bos",
        "sig",
        "spk_emb_random_match",
    ]

    init_sequence_encoder(hparams)

    resample_fn = {}
    for dataset in data_info:
        dataset_output_keys = (
            output_keys
            if dataset == "train"
            else output_keys + ["label_norm_eval"]
        )
        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=dataset_output_keys,
        )

        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False
        if hparams["spk_emb_shuffle"]:
            spk_idx, spk_samplers = group_by_speaker(dynamic_dataset, hparams)
            spk_sample = {}
            spk_emb_random_match_pipeline = partial(
                spk_emb_random_match,
                spk_sample=spk_sample,
                dataset=dynamic_dataset.filtered_sorted(),
            )
            dynamic_dataset.add_dynamic_item(
                func=spk_emb_random_match_pipeline,
                takes=["uttid"],
                provides=["spk_emb_random_match"],
            )
            resample_fn[dataset] = partial(
                resample_spk,
                spk_idx=spk_idx,
                sample=spk_sample,
                dataset=dynamic_dataset,
                spk_samplers=spk_samplers,
            )
            resample_fn[dataset](epoch=0)

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # Exclude samples without phonemes
    if hparams["input"] == "phonemes":
        for key in datasets:
            datasets[key] = datasets[key].filtered_sorted(
                key_test={"phn": lambda value: value}
            )
    datasets["sample"] = select_sample(hparams, datasets)
    return datasets, silence_padding, resample_fn


def select_sample(hparams, datasets):
    """Selects a sample of files for sample generation, freezing the sample if
    requested to persist across multiple experiments

    Arguments
    ---------
    hparams : dict
        experiment hyperparameters
    datasets : dict
        a dictionary of datasets

    Returns
    -------
    dataset : speechbrain.dataio.dataset.FilteredSortedDynamicItemDataset
        the sample dataset
    """
    sample_path = hparams.get("sample_path")
    dataset = None
    if sample_path is not None:
        sample_path = Path(sample_path)
        if sample_path.exists():
            with open(sample_path, "r") as sample_file:
                data_ids = [line.strip() for line in sample_file]
                dataset = FilteredSortedDynamicItemDataset(
                    datasets["valid"], data_ids
                )

    if dataset is None:
        dataset = (
            datasets["valid"]
            .batch_shuffle(1)
            .filtered_sorted(select_n=hparams["num_audio_samples"])
        )
        if sample_path is not None:
            with open(sample_path, "w") as sample_file:
                for data_id in dataset.data_ids:
                    print(data_id, file=sample_file)
    return dataset


def group_by_speaker(dataset, hparams):
    """Groups utterance IDs in a dataset by speaker, for selection. The selection
    is stable based on the seed - calling this method multiple times will always
    result in the same order

    Arguments
    ---------
    dataset : torch.Tensor
        the dataset from which to select items
    hparams : dict
        hyperparameters

    Returns
    -------
    spk_idx : dict
        a str -> int dictionary with a list of utterance indexes
        for every speaker
    spk_samplers : dict
        a reproducible sampler for every speaker
    spk_samplers_it : dict
        an iterator for each sampler
    """
    spk_idx = {}
    spk_samplers = {}
    speakers = []
    generator = torch.Generator()
    generator.manual_seed(hparams["seed"])

    # Group by speaker
    with dataset.output_keys_as(["spk_id"]):
        for idx, item in enumerate(dataset):
            spk_id = item["spk_id"]
            if spk_id not in spk_idx:
                spk_idx[spk_id] = []
            spk_idx[spk_id].append(idx)
            speakers.append(spk_id)

    # Create a reproducible sampler
    for spk_id in speakers:
        sampler = hparams["spk_sampler"](data_source=spk_idx[spk_id])
        spk_samplers[spk_id] = sampler

    return spk_idx, spk_samplers


def resample_spk(sample, spk_idx, spk_samplers, dataset, epoch):
    """Selects new samples

    Arguments
    ---------
    spk_idx : dict
        Data item indexes grouped by speaker
    spk_samplers : dict
        A sampler for each speaker
    spk_samplers_it : dict
        An iterator for each speaker
    epoch : int
        The epoch number

    Returns
    -------
    sample : dict
        a dictionary with uttids as keys and matching
        indexes as values
    """
    if epoch is None:
        epoch = 0
    spk_samplers_it = {}
    for spk_id, sampler in spk_samplers.items():
        sampler.set_epoch(epoch)
        spk_samplers_it[spk_id] = iter(sampler)
    with dataset.output_keys_as(["uttid", "spk_id"]):
        for item in dataset:
            spk_item_idx = next(spk_samplers_it[item["spk_id"]])
            dataset_item_idx = spk_idx[item["spk_id"]][spk_item_idx]
            sample[item["uttid"]] = dataset_item_idx


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


def get_selected_layer_indexes(hparams):
    """Finds the layers of selected layers

    Arguments
    ---------
    hparams : dict
        Hyperparameters
    """
    selected_layers = hparams.get("speech_model_layers")
    available_layers = hparams.get("available_speech_model_layers")
    if not (selected_layers and available_layers):
        return None
    layer_idx = [available_layers.index(layer) for layer in selected_layers]
    return layer_idx


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
    file_name = Path(file_name)
    if not file_name.is_absolute():
        file_name = Path(__file__).parent / "hparams" / file_name
    if not file_name.exists():
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip("\r\n") for line in token_file if line]


def apply_overfit_test(hparams, dataset):
    """Helper for applying an overfit test conditionally based
    on hyperparameters:

    `overfit_test`: whether or not to apply an overfit test
    `overfit_test_sample_count`: the number of samples to use from the
        original dataset
    `overfit_test_epoch_data_count`: the number of samples per epoch

    The function will accept datasets, (train, valid, test) tuples
    or dictionaries of the form:
    {"train": dataset1, "valid": dataset2, "test": dataset3}

    If a tuple or dictionary is used, the training dataset will be of length
    overfit_test_epoch_data_count wheres the evaluation dataset will be of
    length overfit_test_sample_count.

    Arguments
    ---------
    hparams: dict
        parsed hyperparameters
    dataset: DynamicItemDataset|tuple|dict
        One of the following
        a dataset
        a dictionary ({"train": dataset1, "valid": dataset2, "test": dataset3})
        a (train, valid, test)  tuple of datasets

    Returns
    -------
    result: DynamicItemDataset|tuple|dict
        a dataset or collection of datasets suitable for
        an overfitting test - in the same format as the
        dataset argument (single dataset, dictionary and tuple)
    """
    if hparams["overfit_test"]:
        if isinstance(dataset, tuple):
            dataset_train, _, _ = dataset
            dataset_train = apply_overfit_test(hparams, dataset_train)
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = dataset_train, dataset_eval, dataset_eval
        elif isinstance(dataset, dict):
            dataset_train = apply_overfit_test(hparams, dataset["train"])
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            result = {
                "train": dataset_train,
                "valid": dataset_eval,
                "test": dataset_eval,
                "sample": dataset_eval,
            }
        else:
            result = dataset.overfit_test(
                hparams["overfit_test_sample_count"],
                hparams["overfit_test_epoch_data_count"],
            )
    else:
        result = dataset
    return result


RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        yaml = fin.read()

    # Load evaluation hyperparameters
    eval_hparams_file = Path(hparams_file).parent / "eval.yaml"
    if eval_hparams_file.exists():
        logger.info(
            "Using evaluation hyperparameters from %s", eval_hparams_file
        )
        with open(eval_hparams_file) as eval_hparams:
            hparams_yaml = eval_hparams.read()
            yaml = "\n".join([yaml, hparams_yaml])
    else:
        logger.info(
            "%s not found - not using evaluation hyperparameters",
            eval_hparams_file,
        )
    hparams = load_hyperpyyaml(yaml, overrides, overrides_must_match=True)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from libritts_prepare import prepare_libritts

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_libritts,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_json_train": hparams["train_json"],
                "save_json_valid": hparams["valid_json"],
                "save_json_test": (
                    hparams["test_json"]
                    if "test" in hparams["splits"]
                    else None
                ),
                "sample_rate": hparams["sample_rate"],
                "train_split": hparams["train_split"],
                "valid_split": hparams["valid_split"],
                "test_split": (
                    hparams["test_split"]
                    if "test" in hparams["splits"]
                    else None
                ),
                "seed": hparams["seed"],
                "model_name": hparams["model"].__class__.__name__,
            },
        )

    # We can now directly create the datasets for training, valid, and test
    (datasets, silence_padding, resample_fn) = dataio_prepare(hparams)

    # Apply overfit test settings
    datasets = apply_overfit_test(hparams, datasets)
    audio_keys = ["audio_pad", "audio_bos"]

    # Trainer initialization
    tts_brain = TokotronBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    tts_brain.sample_data = datasets["sample"]
    tts_brain.resample_fn = resample_fn

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=use_silence_padding(
            hparams["train_dataloader_opts"], silence_padding, audio_keys
        ),
        valid_loader_kwargs=use_silence_padding(
            hparams["valid_dataloader_opts"], silence_padding, audio_keys
        ),
    )

    # Load best checkpoint for evaluation
    if hparams["testing"]:
        test_summary_file = Path(hparams["output_folder"]) / "eval" / "test" / "summary.json"
        if test_summary_file.exists():
            logging.info("Test run already completed: %s", test_summary_file)
        else:
            test_key_kind = hparams["test_key_kind"]
            test_key = hparams["test_key"]
            eval_kwargs = {
                f"{test_key_kind}_key": test_key
            }
            tts_brain.evaluate(
                test_set=datasets["test"],
                test_loader_kwargs=hparams["test_dataloader_opts"],
                **eval_kwargs
            )
