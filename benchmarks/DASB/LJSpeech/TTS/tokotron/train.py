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
import re
import string
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import clean_padding, clean_padding_
from speechbrain.utils.distributed import run_on_main

base_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(base_dir)

from model.Tokotron import (
    get_silence_token,
    use_silence_padding,
    feature_pad_to,
    RepresentationMode,
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
        audio, audio_length, _, _ = features
        emb = None
        if self.use_spk_emb:
            emb = {"spk": batch.spk_emb.data.squeeze(1)}

        audio = self.transform_audio(audio)
        predictions = self.modules.model(
            input_tokens=tokens,
            input_length=tokens_length,
            audio=audio,
            audio_length=audio_length,
            emb=emb,
        )

        return predictions, features

    def prepare_features(self, batch):
        """Prepares features, depending on the configuration

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation

        Returns
        -------
        audio_bos : torch.Tensor
            Audio features, with BOS
        audio_bos_length : torch.Tensor
            Relative lengths of the audio features, with BOS
        audio_tgt : torch.Tensor
            Target audio features (for loss computation)
        audio_tgt_length : torch.Tensor
            Relative lengths of the target audio features
        """
        if self.representation_mode == RepresentationMode.DISCRETE:
            audio_bos, audio_bos_length = batch.audio_bos
            audio_tgt, audio_tgt_length = batch.audio_pad
            if self.audio_token_offsets is not None:
                audio_bos = torch.cat(
                    [
                        audio_bos[:, : self.hparams.bos_width],
                        audio_bos[:, self.hparams.bos_width :]
                        - self.audio_token_offsets,
                    ],
                    dim=1,
                )
                clean_padding_(audio_bos, audio_bos_length)
                audio_tgt = audio_tgt - self.audio_token_offsets
                clean_padding_(audio_tgt, audio_tgt_length)
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
        return audio_bos, audio_bos_length, audio_tgt, audio_tgt_length

    def get_token_offsets(self):
        """Computes token offsets for tokenizers that require them"""
        token_offsets = None
        if self.hparams.audio_token_offsets:
            token_offsets = (
                torch.arange(
                    self.hparams.audio_tokens_per_step, device=self.device
                )
                * self.hparams.audio_num_tokens
            )[None, None, :]
        return token_offsets

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
        _, _, audio_tgt, audio_tgt_length = features

        audio_tgt = self.transform_audio(audio_tgt)
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
        self.loss_metric = sb.utils.metric_stats.MultiMetricStats(
            metric=self.hparams.compute_cost, batch_eval=True,
        )
        if (
            self.hparams.audio_emb_pretrained
            and epoch == 1
            and stage == sb.Stage.TRAIN
        ):
            vocabulary = None
            if hasattr(self.hparams.token_model, "vocabulary"):
                vocabulary = self.hparams.token_model.vocabulary
            elif hasattr(self.hparams.token_model, "vocabularies"):
                vocabulary = torch.stack(
                    [
                        torch.from_numpy(voc)
                        for voc in self.hparams.token_model.vocabularies
                    ]
                )
            if vocabulary is not None:
                self.modules.model.init_audio_emb(vocabulary)
        # Speaker embeddings are optional and are used only to pretrain
        # multispeaker models
        self.use_spk_emb = getattr(self.hparams, "use_spk_emb", False)

        self.is_evaluating = False
        if self.hparams.eval_enabled:
            if stage == sb.Stage.VALID:
                if self.is_eval_epoch(epoch):
                    self.evaluator.on_evaluate_start(stage, epoch)
                    self.is_evaluating = True
                else:
                    logger.info("No evaluation on epoch %d", epoch)
            elif stage == sb.Stage.TEST:
                self.evaluator.on_evaluate_start(stage, epoch)
                self.is_evaluating = True

        self.audio_token_offsets = self.get_token_offsets()
        self.token_model_kwargs = getattr(
            self.hparams, "token_model_kwargs", {}
        )

        self.transform_audio = getattr(self.hparams, "transform_audio", torch.nn.Identity())

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
        eval_summary_stats = {}
        if stage != sb.Stage.TRAIN and self.is_eval_epoch(epoch):
            self.evaluator.on_evaluate_end()
            eval_summary_stats = self.get_summary_stats()
            stage_stats.update(eval_summary_stats)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            if self.hparams.lr_annealing_mode == "epoch":
                _, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(
                    self.optimizer, new_lr, param_group=0
                )

            lr = self.optimizer.param_groups[0]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.        
            ckpt_kwargs = {
                f"{self.hparams.ckpt_key_kind}_keys": [self.hparams.ckpt_key],
            }
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], **eval_summary_stats},
                num_to_keep=hparams["ckpt_keep"],
                **ckpt_kwargs
            )

    def get_summary_stats(self):
        """Retrieves the stats that needs to be reported on every trial
        in the train log, as indicated in eval_summary_log in eval.yaml

        Returns
        -------
        eval_summary_stats : dict
            A dict with stats"""
        eval_summary = self.evaluator.compute_summary()
        eval_summary_stats = {
            key: eval_summary.get(value)
            for key, value in self.hparams.eval_summary_log.items()
        }
        self._check_threshold(eval_summary_stats)
        return eval_summary_stats

    def _check_threshold(self, eval_summary_stats):
        """Checks threshold values for the defined stats and terminates
        the trials if the parameters are not met. This is necessary because
        some metrics produce bogus high values when the speech samples
        do not contain any speech at all (e.g. UTMOS can be above 3 for
        silence).

        Classic usage: dWER > 0.9 - treat the whole run as "garbage", set
        UTMOS to 0

        Arguments
        ---------
        eval_summary_stats : dict
            Summary statistics
        """
        for key, threshold_value in self.hparams.eval_threshold.items():
            key, threshold_type = key.split("_")
            value = eval_summary_stats[key]
            if threshold_type == "min":
                meets = value >= threshold_value
            elif threshold_type == "max":
                meets = value <= threshold_value
            else:
                raise ValueError(
                    f"Invalid threshold definition: {key}, check eval_threshold"
                )
            if not meets:
                eval_summary_stats["broken"] = True
                for key, value in self.hparams.eval_threshold_set.items():
                    eval_summary_stats[key] = value

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        loss = super().fit_batch(batch)
        if self.hparams.lr_annealing_mode == "step":
            self.hparams.lr_annealing(self.optimizer)
        return loss

    def init_optimizers(self):
        """Custom optimizer initialization
        """
        if self.representation_mode == RepresentationMode.CONTINUOUS:
            audio_emb_params = self.modules.model.decoder.audio_emb.parameters()
            audio_emb_params_set = set(audio_emb_params)
            model_params = [
                param
                for param in self.modules.parameters()
                if param not in audio_emb_params_set
            ]
            self.optimizer = self.opt_class(
                [
                    {"params": model_params},
                    {
                        "params": audio_emb_params,
                        "lr": self.hparams.audio_emb_lr,
                        "weight_decay": self.hparams.audio_emb_weight_decay,
                    },
                ]
            )
        else:
            self.optimizer = self.opt_class(
                self.modules.model.parameters(), lr=self.hparams.lr
            )

    def create_waveform(self, audio, length):
        """Creates a waveform from a discrete or continuous audio
        representation

        Arguments
        ---------
        audio : torch.Tensor
            An audio tensor (Batch x Length x Heads or Batch x Length x Heads x Features)
        lengths : torch.Tensor
            A 1-D tensor

        Returns
        -------
        wav : torch.Tensor
        """
        self.modules.tokenizer.device = self.device
        if hasattr(self.modules.tokenizer, "codec_vocoder"):
            self.modules.tokenizer.codec_vocoder.to(self.device)
            self.modules.tokenizer.codec_vocoder.device = self.device
        with torch.no_grad():
            if self.audio_token_offsets is not None:
                audio = clean_padding(audio + self.audio_token_offsets, length)
            wav = self.modules.tokenizer.tokens_to_sig(
                audio, **self.token_model_kwargs
            )
            wav = clean_padding(wav, length)
            wav = wav.to(self.device)
        return wav

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


INPUT_FEATURE_MAP = {"text": "label_norm", "phonemes": "phonemes"}


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

    representation_mode = RepresentationMode(hparams["representation_mode"])

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    label_encoder = hparams["label_encoder"]
    input_feature = INPUT_FEATURE_MAP[hparams["input"]]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_norm", "label_norm_eval")
    def text_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        label_norm = label.upper()
        yield label_norm

        label_norm_eval = RE_PUNCTUATION.sub("", label_norm)
        yield label_norm_eval

    @sb.utils.data_pipeline.takes(input_feature)
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(label):
        """Processes the transcriptions to generate proper labels"""
        return label_encoder.encode_sequence_torch(label)

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

    use_silence_padding = hparams.get("use_silence_padding", True)

    if representation_mode == RepresentationMode.DISCRETE:
        model_key = "tokenizer"
    else:
        model_key = "ssl_model"

    audio_tokens_per_step = hparams["audio_tokens_per_step"]
    if (
        use_silence_padding
        and representation_mode == RepresentationMode.DISCRETE
    ):
        silence_token = get_silence_token(
            hparams[model_key],
            num_codebooks=(
                hparams["speech_model_layers"]
                if "speech_model_layers" in hparams
                else audio_tokens_per_step
            )
        )
        if silence_token.dim() == 2:
            silence_token = silence_token.squeeze(-1)
    else:
        silence_token = (
            torch.ones(hparams["audio_tokens_per_step"], dtype=torch.int64)
            * hparams["eos_index"]
        )
    silence_padding = silence_token.cpu()
    silence_padding = silence_padding[:audio_tokens_per_step]
    silence_padding_len = int(math.ceil(hparams["silence_padding"]))
    bos_width = hparams.get("bos_width", 1)
    audio_bos_prefix = (
        torch.ones(bos_width, audio_tokens_per_step) * hparams["bos_index"]
    )

    tokens_loader = hparams.get("tokens_loader")
    if "speech_model_layers" in hparams:
        tokens_loader_kwargs = {
            "num_codebooks": get_selected_layer_indexes(hparams)
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

    dynamic_items = [
        text_pipeline,
        tokens_pipeline,
        audio_pipeline,
        audio_ref_pipeline,
    ]

    init_sequence_encoder(hparams)
    output_keys = [
        "uttid",
        "tokens",
        "label_norm_eval",
    ]
    if representation_mode == RepresentationMode.DISCRETE:
        output_keys += [
            "audio_pad",
            "audio_bos",
        ]
    else:
        output_keys.append("sig")

    eval_output_keys = [*output_keys, "sig"]
    for dataset in data_info:
        if dataset == "train":
            dataset_output_keys = output_keys
        else:
            dataset_output_keys = eval_output_keys

        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=dataset_output_keys,
        )

        datasets[dataset] = dynamic_dataset
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

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
    data_scale = hparams.get("data_scale")
    if data_scale:
        scaled_data_count = int(len(datasets["train"]) * data_scale)
        datasets["train"] = datasets["train"].filtered_sorted(
            select_n=scaled_data_count
        )

    return datasets, silence_padding


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
            dataset_train, dataset_valid, _ = dataset
            dataset_train = apply_overfit_test(hparams, dataset_train)
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            dataset_eval.set_output_keys(
                list(dataset_valid.pipeline.output_mapping.keys())
            )
            result = dataset_train, dataset_eval, dataset_eval
        elif isinstance(dataset, dict):
            dataset_train = apply_overfit_test(hparams, dataset["train"])
            dataset_eval = dataset_train.filtered_sorted(
                select_n=hparams["overfit_test_sample_count"]
            )
            dataset_eval.set_output_keys(
                list(dataset["valid"].pipeline.output_mapping.keys())
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
    if not eval_hparams_file.exists():
        eval_hparams_file = Path(__file__).parent / "hparams" / "eval.yaml"
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

    from ljspeech_prepare import prepare_ljspeech

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_ljspeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["prepare_save_folder"],
                "splits": hparams["splits"],
                "split_ratio": hparams["split_ratio"],
                "seed": hparams["seed"],
                "extract_phonemes": hparams["input"] == "phonemes",
                "model_name": "tokotron",
                "g2p_src": hparams["g2p_src"],
                "skip_ignore_folders": hparams["prepare_skip_ignore_folders"],
                "frozen_split_path": hparams.get("frozen_split_path"),
                "device": run_opts.get("device", "cpu"),
            },
        )

    # We can now directly create the datasets for training, valid, and test
    datasets, silence_padding = dataio_prepare(hparams)

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

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    dataloader_opts = [
        hparams[f"{key}_dataloader_opts"] for key in ["train", "valid", "test"]
    ]
    representation_mode = RepresentationMode(hparams["representation_mode"])
    if representation_mode == RepresentationMode.DISCRETE:
        dataloader_opts = [
            use_silence_padding(opts, silence_padding, audio_keys)
            for opts in dataloader_opts
        ]
    (
        train_dataloader_opts,
        valid_dataloader_opts,
        test_dataloader_opts,
    ) = dataloader_opts

    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Load best checkpoint for evaluation
    if hparams["testing"]:
        test_summary_file = Path(hparams["output_folder"]) / "eval" / "test" / "summary.json"
        if test_summary_file.exists():
            logging.info("Test run already completed: %s", test_summary_file)
        else:
            test_summary_file = Path(hparams["output_folder"]) / "eval" / "test" / "summary.json"
            if test_summary_file.exists():
                logging.info("Test run already completed: %s", test_summary_file)
            else:
                test_key_kind = hparams["test_key_kind"]
                test_key = hparams["test_key"]
                if test_key:
                    eval_kwargs = {
                        f"{test_key_kind}_key": test_key
                    }
                tts_brain.evaluate(
                    test_set=datasets["test"],
                    test_loader_kwargs=hparams["test_dataloader_opts"],
                    **eval_kwargs
                )


    # Save final checkpoint (fixed name)
    tts_brain.checkpointer.save_checkpoint(name="latest")
