""" Specifies the inference interfaces for speech quality
evaluation, used to assess the quality/intelligibility of
text-to-speech systems

Authors:
* Artem Ploujnikov 2024
"""

from speechbrain.inference.interfaces import Pretrained
from speechbrain.lobes.models.huggingface_transformers import Whisper
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.utils.data_utils import pad_right_to
from speechbrain.utils.fetching import fetch
from collections import namedtuple
from pathlib import Path
from torch import nn
import torch
import torchaudio
import re
import string
import logging


logger = logging.getLogger(__name__)


has_transformers = False
try:
    from transformers import AutoModelForAudioXVector

    has_transformers = True
except ImportError:
    logger.warning(
        "transformers library not found - some evaluators may be disabled"
    )


RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


SAMPLE_RATE = 16000
DEFAULT_ENCODER_HUB = "chaanks/wav2vec2-small"
DEFAULT_MODEL_URL = "https://huggingface.co/chaanks/UTMOS/resolve/main"
DEFAULT_MODEL_NAME = "utmos.ckpt"
DEFAULT_SAVE_DIR = "./pretrained_models"
DEFAULT_JUDGE_ID = 288
DEFAULT_DOMAIN_ID = 0

SpeechEvaluationResult = namedtuple(
    "SpeechEvaluationResult", ["score", "details"]
)


class SpeechEvaluator:
    """A base class for speech evaluators

    Arguments
    ---------
    sample_rate : int
        The audio sample rate this evaluator expects
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def evaluate_file(self, file_name, text=None):
        """Evaluates a single file

        Arguments
        ---------
        file_name : str|pathlib.Path
            The file name to evaluate
        text : str
            The ground truth text, if applicable

        Returns
        -------
        result: SpeechEvaluationResult
            the evaluation result
        """
        wav = self.read_audio(str(file_name)).to(self.device)
        result = self.evaluate(
            wavs=wav.unsqueeze(0),
            length=torch.ones(1).to(self.device),
            text=[text],
        )
        return SpeechEvaluationResult(
            score=result.score.item(),
            details={
                key: _unbatchify(value) for key, value in result.details.items()
            },
        )

    def evaluate_files(self, file_names, text=None):
        """Evaluates multiple files

        Arguments
        ---------
        file_names : list
            A list of files

        text : list
            File transcripts (not required for all evaluators)

        Returns
        -------
        result : list
            a list of SpeechEvaluationResult instances
        """
        if text is None:
            text = [None] * len(file_names)
        items = [
            {"wav": self.read_audio(str(file_name)), "text": item_text}
            for file_name, item_text in zip(file_names, text)
        ]
        batch = PaddedBatch(items)
        return self.evaluate(
            wavs=batch.wav.data.to(self.device),
            length=batch.wav.lengths.to(self.device),
            text=batch.text,
        )

    def read_audio(self, file_name):
        """Reads an audio file, resampling if necessary

        Arguments
        ---------
        file_name : str | path-like
            The file path

        Returns
        -------
        audio : torch.Tensor
            the audio
        """
        audio, audio_sample_rate = torchaudio.load(str(file_name))
        return self.resample(audio, audio_sample_rate)

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        wavs_length_ref=None,
        sample_rate=None,
    ):
        """Evaluates samples

        Arguments
        ---------
        wavs : torch.Tensor
            the waveforms to evaluate

        length : torch.Tensor
            relative lengths (a 1-D tensor)

        text : list
            Evaluator-specific metadata

        wavs_ref : torch.Tensor
            the reference waveforms

        wavs_length_ref
            the reference waveform lengths

        sample_rate: int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        Returns
        -------
        result : list
            A list of SpeechEvaluationResult objects,
            one for each sample"""
        raise NotImplementedError()

    def resample(self, audio, sample_rate=None):
        """Resamples the audio, if necessary

        Arguments
        ---------
        audio : torch.Tensor
            the audio to be resampled
        sample_rate : int
            the sample rate of the audio

        Returns
        -------
        audio : torch.Tensor
            the target audio, resampled if necessary
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sample_rate, new_freq=self.sample_rate
            )
        return audio


def _unbatchify(value):
    """Removes the batch dimension from the tensor. If a single
    number is returned in any shape, the function converts
    the result to a numeric value. Values that are not tensors
    are returned unmodified

    Arguments
    ---------
    value : object
        the value

    Returns
    -------
    value : object
        the value with the batch dimension removed, if applicable
    """
    if torch.is_tensor(value):
        if value.dim() == 0 or not any(dim > 1 for dim in value.shape):
            value = value.item()
        else:
            value = value.squeeze(0)
    return value


class SpeechEvaluationRegressionModel(Pretrained):
    """A pretrained wrapper for regression-based evaluaton
    models"""

    def __call__(self, wavs, length):
        return self.mods.model(wavs, length)


class ASRSpeechEvaluator(SpeechEvaluator):
    """A superclass for ASR speech evaluators"""

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        """Evaluates samples

        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate

        length: torch.Tensor
            relative lengths (a 1-D tensor)

        text : list, optional
            Ground truth text

        wavs_ref : torch.Tensor
            the reference waveforms

        length_ref : torch.Tensor
            the reference waveform lengths


        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        sample_rate_ref : int, optional
            The sample rate of the reference samples

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        details = self.evaluate_samples(
            wavs=wavs, length=length, text=text, sample_rate=sample_rate
        )
        if wavs_ref is not None:
            details_ref = self.evaluate_samples(
                wavs=wavs_ref,
                length=length_ref,
                text=text,
                sample_rate=sample_rate_ref,
            )
            details.update(
                {f"{key}_ref": value for key, value in details_ref.items()}
            )
            # Redundant: it is the same
            del details["target_ref"]
            details.update(self.compute_diff_rate(details, device=wavs.device))

        return SpeechEvaluationResult(score=details["wer"], details=details,)

    def compute_diff_rate(self, details, device):
        """Computes the differential token rate

        Arguments
        ---------
        details : dict
            The evaluation details
            Keys:
                "pred": ASR predictions for the TTS sample
                "pred_ref": ASR predictions for the ground
                truth

        Returns
        -------
        result: dict
            A dictionary with the following keys

            dwer : torch.Tensor
                The differential Word Error Rate (dWER)
            dcer : torch.Tensor
                The differential Character Error Rate (dCER)

        """
        ids = range(1, len(details["pred"]) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        pred = self._replace_blanks(details["pred"])
        pred_ref = self._replace_blanks(details["pred_ref"])
        wer_metric.append(ids, pred, pred_ref)
        cer_metric.append(ids, pred, pred_ref)
        dwer = torch.tensor(
            [score["WER"] for score in wer_metric.scores], device=device
        )
        dcer = torch.tensor(
            [score["WER"] for score in cer_metric.scores], device=device
        )
        return {"dwer": dwer, "dcer": dcer}

    def _replace_blanks(self, preds):
        """Replaces blanks with single spaces, preventing an exception
        in the case of an unintelligible sample

        Arguments
        ---------
        """
        return [" " if item == "" else item for item in preds]


class WhisperASRSpeechEvaluator(ASRSpeechEvaluator):
    """A speech evaluator implementation based on Whisper ASR

    Arguments
    ---------
    source : str
        The source directory
    savedir : str, optional
        The path where Whisper will be saved
    sample_rate: int, optional
        The audio sample rate
    min_decode_ratio : float, optional
        The minimum decode ratio
    max_decode_ratio : float, optional
        The maximum decode ratio
    run_opts : dict, optional
        Run options for the Whisper model
    unbatch : bool, optional
        If enabled, which is the default, the implementation
        will evaluate samples one by one with a batch size of
        1 and then "reassemble" the original batch. This is
        sometimes needed because batched inference has been
        found to result in decreased performance, primarily
        due to masks not being applied to convolutional layers
    """

    def __init__(
        self,
        source,
        savedir=None,
        sample_rate=22050,
        min_decode_ratio=0.0,
        max_decode_ratio=1.0,
        run_opts=None,
        unbatch=True,
    ):
        super().__init__(sample_rate=sample_rate)
        if run_opts is None:
            run_opts = {}
        if savedir is None:
            savedir = "."
        self.model = Whisper(
            source, savedir, sample_rate, freeze=True, freeze_encoder=True,
        )
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.searcher = S2SWhisperGreedySearcher(
            self.model,
            min_decode_ratio=min_decode_ratio,
            max_decode_ratio=max_decode_ratio,
        )
        device = run_opts.get("device", next(self.model.parameters()).device)
        self.unbatch = unbatch
        self.to(device)

    def evaluate_samples(self, wavs, length, text, sample_rate):
        """Evaluates a batch of samples

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list
            Text labels corresponding to the waveforms
        sample_rate : int
            The sample rate of the waveforms

        Returns
        -------
        results : dict
            The evaluation results
        """
        if self.unbatch:
            batch_size = len(wavs)
            length_abs = (length * wavs.size(1)).int()
            results = [
                self._evaluate_samples(
                    wavs[idx : idx + 1, : length_abs[idx].item()],
                    torch.ones(1, device=wavs.device),
                    text[idx : idx + 1],
                    sample_rate,
                )
                for idx in range(batch_size)
            ]
            result = {
                "wer": torch.stack(
                    [result["wer"] for result in results]
                ).squeeze(-1),
                "cer": torch.stack(
                    [result["cer"] for result in results]
                ).squeeze(-1),
                "pred": [result["pred"][0] for result in results],
                "target": text,
            }
            return result
        else:
            return self._evaluate_samples(wavs, length, text, sample_rate)

    def _evaluate_samples(self, wavs, length, text, sample_rate):
        """Evaluates a batch of samples. This function is meant
        to be used internally. evaluate_samples will call
        it multiple times if unbatch is enabled.

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list
            Text labels corresponding to the waveforms
        sample_rate : int
            The sample rate of the waveforms

        Returns
        -------
        results : dict
            The evaluation results
        """
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        wavs = self.resample(wavs, sample_rate)
        wavs = self.model.pad_or_trim(wavs)
        mels = self.model.log_mel_spectrogram(wavs)
        enc_out = self.model.forward_encoder(mels)
        predicted_words, _, _, _ = self.searcher(enc_out.detach(), length)
        predicted_words = self.model.tokenizer.batch_decode(
            predicted_words, skip_special_tokens=True
        )
        predicted_words = [self.normalize(text) for text in predicted_words]
        ids = range(1, len(wavs) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        wer_metric.append(ids, predicted_words, text)
        cer_metric.append(ids, predicted_words, text)
        wer = torch.tensor(
            [score["WER"] for score in wer_metric.scores], device=wavs.device
        )
        cer = torch.tensor(
            [score["WER"] for score in cer_metric.scores], device=wavs.device
        )
        return {
            "wer": wer,
            "cer": cer,
            "pred": predicted_words,
            "target": text,
        }

    def normalize(self, text):
        """Performs text normalization (uppercase, remove whitespace,
        remove punctuation)

        Arguments
        ---------
        text : str
            Unnormalized text

        Returns
        -------
        text : str
            Normalized text
        """
        text = text.upper()
        text = text.strip()
        text = RE_PUNCTUATION.sub("", text)
        return text

    def to(self, device):
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.Device
            the target device
        """
        self.model = self.model.to(device)
        return self


def itemize(result):
    """Converts a single batch result into per-item results

    Arguments
    ---------
    result: SpeechEvaluationResult
        a single batch result

    Returns
    -------
    results: list
        a list of individual SpeechEvaluationResult instances"""

    return [
        SpeechEvaluationResult(
            score=result.score[idx],
            details={key: value[idx] for key, value in result.items()},
        )
        for idx in range(len(result.score))
    ]


def init_asr_metrics():
    """Initializes the WER and CER metrics

    Returns
    -------
    wer_metric : ErrorRateStats
        the Word Error Rate (WER) metric
    cer_metric : ErrorRateStats
        the Character Error Rate (CER) metric"""
    wer_metric = ErrorRateStats()
    cer_metric = ErrorRateStats(split_tokens=True)
    return wer_metric, cer_metric


class BulkSpeechEvaluator:
    """A base class for a speech evaluator that is invoked for a series of filesystem files
    rather than one batch at a time. This is useful for implementing wrappers around
    command-line tools that would be impractical to run for each batch because of
    long initialization time (to load models, etc)"""

    def evaluate_files(self, file_names, text=None, file_names_ref=None):
        """Evaluates multiple files

        Arguments
        ---------
        file_names : list
            A list of files

        text : list, optional
            File transcripts (not required for all evaluators)

        file_names_ref : list, optional
            A list of reference files / ground truths (if applicable)

        Returns
        -------
        result : SpeechEvaluationResult
            a consolidated evaluation result
        """
        raise NotImplementedError()


class UTMOSModel(nn.Module):
    """The UTMOS model wrapper

    Arguments
    ---------
    source : str
        The WavLM source
    save_path : str | path-like
        The path where the model will be saved
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    multiplier : float, optional
        The number that the raw model output is multiplied by
        to compute the score
    offset : float, optional
        The number that (raw output * multiplier) will be added
        to in order to get the score
    """

    def __init__(
        self,
        source,
        save_path,
        features_dim=768,
        num_domains=3,
        domain_dim=128,
        num_judges=3000,
        judge_dim=128,
        decoder_hidden_size=512,
        multiplier=2.0,
        offset=3.0,
    ):
        super().__init__()

        self.ssl_encoder = Wav2Vec2(
            source,
            save_path,
            freeze=True,
            output_norm=False,
            freeze_feature_extractor=True,
            output_all_hiddens=False,
        )

        self.domain_embedding = nn.Embedding(num_domains, domain_dim)
        self.judge_embedding = nn.Embedding(num_judges, judge_dim)

        self.decoder = nn.LSTM(
            input_size=features_dim + domain_dim + judge_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(decoder_hidden_size * 2, 2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        )
        self.multiplier = multiplier
        self.offset = offset

    def forward(self, wav, domain_id=None, judge_id=None):
        """Computes the forward pass

        Arguments
        ---------
        wav : torch.Tensor
            The raw waveforms
        domain_id : torch.Tensor
            The domain identifiers
        judge_id : torch.Tensor
            The judge identifier

        Returns
        -------
        result : torch.Tensor
            The predicted rating(s)
        """

        if domain_id is None:
            domain_id = torch.zeros(
                len(wav), dtype=torch.int, device=wav.device
            )
        if judge_id is None:
            judge_id = (
                torch.ones(len(wav), dtype=torch.int, device=wav.device)
                * DEFAULT_JUDGE_ID
            )

        ssl_features = self.ssl_encoder(wav)
        domain_emb = self.domain_embedding(domain_id)
        judge_emb = self.judge_embedding(judge_id)

        domain_emb = domain_emb.unsqueeze(1).expand(
            -1, ssl_features.size(1), -1
        )
        judge_emb = judge_emb.unsqueeze(1).expand(-1, ssl_features.size(1), -1)
        concatenated_feature = torch.cat(
            [ssl_features, domain_emb, judge_emb], dim=2
        )

        decoder_output, _ = self.decoder(concatenated_feature)
        pred = self.classifier(decoder_output)

        return pred.mean(dim=1).squeeze(1) * self.multiplier + self.offset


class UTMOSSpeechEvaluator(SpeechEvaluator):
    """The UTMOS speech evaluator wrapper

    Github: https://github.com/sarulab-speech/UTMOS22
    HuggingFace: https://huggingface.co/spaces/sarulab-speech/UTMOS-demo


    Arguments
    ---------
    source : str, optional
        The WavLM source
    save_path : str | path-like, optional
        The path where the model will be saved
    model_name : str
        The name of the model hub
    model_url : str
        The model URL (if applicable)
    domain_id : int
        The domain ID of the underlying model
    judge_id : int
        The judge ID to use (given UTMOS was trained as an ensemble
        of judges)
    run_opts: dict, optional
        The run options
    sample_rate : int
        The sample rate of the underlying model
    """

    def __init__(
        self,
        source=None,
        save_path=None,
        model_name=None,
        model_url=None,
        domain_id=None,
        judge_id=None,
        run_opts=None,
        sample_rate=16000,
    ):
        super().__init__(sample_rate=sample_rate)
        self.model = UTMOSModel(source=source, save_path=save_path,)
        if run_opts is not None:
            device = run_opts.get("device")
            if device:
                self.model = self.model.to(device)
        fetch(model_name, model_url, save_path)
        model_path = Path(save_path) / model_name
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.domain_id = domain_id
        self.judge_id = judge_id

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        """Evaluates a batch of waveforms using UTMOS

        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate
        length: torch.Tensor
            relative lengths (a 1-D tensor)
        text : list, optional
            Ground truth text. Ignored for UTMOS.
        wavs_ref : torch.Tensor
            the reference waveforms. Ignored for UTMOS.
        length_ref : torch.Tensor
            the reference waveform lengths. Ignored for UTMOS.
        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model
        sample_rate_ref : int, optional
            The sample rate of the reference samples. Ignored for UTMOS.

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        wavs = self.resample(wavs, sample_rate=sample_rate)
        domain_id, judge_id = None, None
        if self.domain_id is not None:
            domain_id = (
                torch.ones(len(wavs), device=wavs.device) * self.domain_id
            )
        if self.judge_id is not None:
            judge_id = torch.ones(len(wavs), device=wavs.device) * self.judge_id

        scores = self.model(wav=wavs, domain_id=domain_id, judge_id=judge_id)
        return SpeechEvaluationResult(score=scores, details={"utmos": scores})


class SpkSimWavLM(SpeechEvaluator):
    """A speaker similarity evaluator based on WavLM / XVector

    Arguments
    ---------
    source : str
        The model hub to use
    savedir : str
        The path where the model will be saved
    model_sample_rate : int, optional
        The sample rate to which all samples will be resampled
        before being processed
    """

    def __init__(
        self,
        source,
        savedir,
        model_sample_rate=16000,
        run_opts=None,
        *args,
        **kwargs,
    ):
        if not has_transformers:
            raise ValueError(
                "Unable to use the SpkSimWavLM evaluator because the "
                "transformers library is not enabled"
            )
        if run_opts is None:
            run_opts = {}
        device = run_opts.get("device")
        self.model = AutoModelForAudioXVector.from_pretrained(
            source, cache_dir=savedir, *args, **kwargs
        )
        if device is not None:
            self.model = self.model.to(device)

        self.model.eval()
        self.model_sample_rate = model_sample_rate
        self.device = next(self.model.parameters()).device

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        # Resample
        if sample_rate is not None:
            wavs = torchaudio.functional.resample(
                wavs, orig_freq=sample_rate, new_freq=self.model_sample_rate
            )
        if sample_rate_ref is not None:
            wavs_ref = torchaudio.functional.resample(
                wavs_ref,
                orig_freq=sample_rate_ref,
                new_freq=self.model_sample_rate,
            )

        # Concatenate
        batch_size, wavs_max_len = wavs.shape
        _, wavs_ref_max_len = wavs_ref.shape
        length_abs = length * wavs_max_len
        length_ref_abs = length_ref * wavs_ref_max_len
        max_len = max(wavs_max_len, wavs_ref_max_len)
        wavs, _ = pad_right_to(wavs, (batch_size, max_len))
        wavs_ref, _ = pad_right_to(wavs_ref, (batch_size, max_len))
        audio = torch.cat([wavs, wavs_ref])

        length_cat_abs = torch.cat([length_abs, length_ref_abs])
        # Attention mask
        attention_mask = length_to_mask(
            length_cat_abs.int()
        ).long()  # 0 for masked tokens
        # Forward
        with torch.inference_mode():
            embs = self.model(
                input_values=audio,
                attention_mask=attention_mask,
                output_attentions=False,
            ).embeddings
        hyp_embs, ref_embs = embs.split([len(wavs), len(wavs_ref)])
        scores = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )

        return SpeechEvaluationResult(scores, {"score": scores})


def vocoder_to_device(vocoder, device):
    """A fix for vocoders that do not properly handle
    the .to() function and require the device to be set manually

    Arguments
    ---------
    vocoder : torch.nn.Module
        a vocoder
    device : str | torch.Device
        the target device
    """
    if hasattr(vocoder, "model") and hasattr(vocoder.model, "device"):
        vocoder.model.device = device
    elif hasattr(vocoder, "device"):
        vocoder.device = device


class Tracker:
    """A tracker that makes it possible to resume evaluation

    Arguments
    ---------
    file_name : str | path-like
        The path to the tracker file"""

    def __init__(self, file_name):
        self.file_name = Path(file_name)

    def mark_processed(self, item_id):
        """Marks the specified file as processed

        Arguments
        ---------
        item_id : str|enumerable
            The item ID or a list of IDS
        """
        if isinstance(item_id, str):
            item_id = [item_id]
        with open(self.file_name, "a+") as tracker_file:
            for item in item_id:
                print(item, file=tracker_file)

    def filter(self, dataset):
        """Filters a dataset using the tracker file

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            A dataset

        Returns
        -------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            The dataset, possibly filtered
        """
        if self.file_name.exists():
            with open(self.file_name) as tracker_file:
                processed_ids = set(line.strip() for line in tracker_file)
                remaining_ids = [
                    data_id
                    for data_id in dataset.data_ids
                    if data_id not in processed_ids
                ]
                logger.info(
                    "Tracker %s already exists, %d items already processed, %d items remaining",
                    self.file_name,
                    len(processed_ids),
                    len(remaining_ids),
                )
                dataset = FilteredSortedDynamicItemDataset(
                    dataset, remaining_ids
                )
        else:
            logger.info(
                "Tracker %s does not exist, evaluating from the beginning"
            )
        return dataset

    def get_processed(self):
        """Retrieves the IDs of items that have been processed

        Returns
        -------
        processed_ids : list
            The list of file IDs
        """
        if self.file_name.exists():
            with open(self.file_name, "r") as tracker_file:
                processed_ids = [line.strip() for line in tracker_file]
        else:
            processed_ids = []
        return processed_ids


class CommandError(Exception):
    """Thrown when an external command returns an error

    Arguments
    ---------
    cmd : str
        The command that was run
    output : str
        The captured standard output stream
    err : str
        The captured standard error stream
    return_code : int
        The return code"""

    def __init__(self, cmd, output, err, return_code):
        super().__init__(
            f"Command {cmd} returned code {return_code}\n"
            f"Output: {output}\n"
            f"Errors: {err}"
        )
        self.cmd = cmd
        self.output = output
