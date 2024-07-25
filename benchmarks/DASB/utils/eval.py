""" Specifies the inference interfaces for speech quality
evaluation, used to assess the quality/intelligibility of
text-to-speech systems

Authors:
* Artem Ploujnikov 2024
"""

from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.lobes.models.huggingface_transformers import Whisper
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.utils.superpowers import run_shell
from collections import namedtuple
from pathlib import Path
import os
import torch
import torchaudio
import re
import string
import logging
import shutil
import shlex
import subprocess

logger = logging.getLogger(__name__)

RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


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


class RegressionModelSpeechEvaluator(SpeechEvaluator):
    """A speech evaluator that uses a regression model
    that produces a quality score (e.g. SSL fine-tuning)
    for a sample of speech

    Arguments
    ---------
    source : str
        The source model path or HuggingFace hub name
    sample_rate : int
        The audio sample rate this evaluator expects
    """

    def __init__(self, source, sample_rate=None, *args, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self.model = SpeechEvaluationRegressionModel.from_hparams(
            source, *args, **kwargs
        )

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
        """Evaluates a batch of waveforms

        Arguments
        ---------
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
        wavs = self.resample(wavs, sample_rate)
        scores = self.model(wavs, length)
        while scores.dim() > 1 and scores.size(-1) == 1:
            scores = scores.squeeze(-1)
        return SpeechEvaluationResult(score=scores, details={"score": scores})


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


class EncoderDecoderASRSpeechEvaluator(ASRSpeechEvaluator):
    """A speech evaluator implementation based on ASR.
    Computes the Word Error Rate (WER), Character Error Rate (CER)
    and a few other metrics

    Arguments
    ---------
    sample_rate : int
        The audio sample rate this evaluator expects
    """

    def __init__(self, source, sample_rate=None, *args, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self.asr = EncoderDecoderASR.from_hparams(source, *args, **kwargs)
        self.device = next(self.asr.mods.parameters()).device

    def evaluate_samples(self, wavs, length, text, sample_rate):
        wavs = self.resample(wavs, sample_rate)
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        predicted_words, scores, log_probs = self.transcribe_batch_with_details(
            wavs, length
        )
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
        prob_mean = log_probs.exp().mean(dim=-1)
        return {
            "wer": wer,
            "cer": cer,
            "beam_score": scores,
            "prob_mean": prob_mean,
            "pred": predicted_words,
            "target": text,
        }

    def transcribe_batch_with_details(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        predicted_words : list
            The raw ASR predictions, fully decoded
        best_scores : list
            The best scores (from beam search)
        best_log_probs : list
            The best predicted log-probabilities (from beam search)


        Returns
        -------
        predicted_words : list
            The predictions

        best_scores : torch.Tensor
            The best scores (from beam search)

        best_log_probs : torch.Tensor
            The best log-probabilities

        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.asr.encode_batch(wavs, wav_lens)
            (
                hyps,
                best_lens,
                best_scores,
                best_log_probs,
            ) = self.asr.mods.decoder(encoder_out, wav_lens)
            predicted_words = [
                self.asr.tokenizer.decode_ids(token_seq) for token_seq in hyps
            ]
        return predicted_words, best_scores, best_log_probs

    def to(self, device):
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.Device
            the target device
        """
        self.asr = self.asr.to(device)
        return self


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


UTMOS_REPO = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo"


class UTMOSSpeechEvaluator(BulkSpeechEvaluator):
    """An evaluation wrapper for UTMOS

    Github: https://github.com/sarulab-speech/UTMOS22
    HuggingFace: https://huggingface.co/spaces/sarulab-speech/UTMOS-demo

    Arguments
    ---------
    model_path : str | path-like
        The path where the HuggingFace repository was extracted
    output_folder : str | path-like
        The folder where results will be output
    ckpt_path : str | path-like
        The path to the checkpoint to be used
    script : str | path-like
        The path to the evaluation script, defaults to the bundled
        predict.py
    python : str | path-like, optional
        The path to the Python interpreter to be used, defaults to
        "python". Depending on the environment, it might need to be
        changed (e.g. to "python3" or an absolute path to the interpreter)
    use_python : bool
        Whether to launch the script using python. This flag will need to be
        set to False in environments where running UTMOS requires a wrapper shell
        script (e.g. to initialize a different Python virtual environment from
        the one in which SpeechBrain is running)
    tmp_folder : str | path-like, optional
        The temporary folder where files will be copied for evaluation. If
        omitted, it will be set to output_folder. This can be useful on
        compute environments that provide fast local storage (e.g. certain
        compute clusters)
    repo : str
        The repor
    """

    def __init__(
        self,
        model_path,
        output_folder,
        ckpt_path,
        script="predict.py",
        python="python",
        use_python=True,
        batch_size=8,
        tmp_folder=None,
        repo=UTMOS_REPO,
    ):
        self.output_folder = Path(output_folder)
        rand = torch.randint(1, 999999999, (1,)).item()
        if tmp_folder is None:
            tmp_folder = self.output_folder
        else:
            tmp_folder = Path(tmp_folder)
        self.eval_path = (tmp_folder / f"eval_{rand}").absolute()
        self.model_path = Path(model_path).absolute()
        script = self.model_path / script
        self.script = script
        self.ckpt_path = Path(ckpt_path).absolute()
        self.batch_size = batch_size
        self.python = python
        self.use_python = use_python
        self.repo = repo
        self.install()

    def install(self):
        if self.model_path.exists():
            logger.info("UTMOS is already installed in %s", self.model_path)
            return
        logger.info(
            "Attempting to install UTMOS from %s to %s",
            self.repo,
            self.model_path,
        )
        cmd = shlex.join(
            [
                "git",
                "-C",
                str(self.model_path.parent),
                "clone",
                self.repo,
                str(self.model_path.name),
            ]
        )
        output, err, return_code = run_shell(cmd)
        if return_code != 0:
            raise CommandError(cmd, output, err, return_code)
        logger.info("Repository clone successful, performing an LFS fetch")
        cwd = Path.cwd()
        try:
            os.chdir(self.model_path)
            cmd = shlex.join(["git", "lfs", "fetch"])
            output, err, return_code = run_shell(cmd)
            if return_code != 0:
                raise CommandError(cmd, output, err, return_code)
        finally:
            os.chdir(cwd)
        if not self.ckpt_path.exists():
            raise ValueError("ckpt_path {ckpt_path} does not exist")

    def evaluate_files(self, file_names, text, file_names_ref=None):
        """Evaluates multiple files

        Arguments
        ---------
        file_names : list
            A list of files

        text : list
            File transcripts (not required for all evaluators)
            Not used in this evaluator

        file_names_ref : list, optional
            A list of reference files / ground truths (if applicable)
            Not used in this evaluator

        Returns
        -------
        result : SpeechEvaluationResult
            a consolidated evaluation result
        """
        current_path = os.getcwd()
        try:
            self.eval_path.mkdir(parents=True, exist_ok=True)
            logger.info("Copying the files to '%s'", self.eval_path)
            for file_name in file_names:
                target_file_name = self.eval_path / Path(file_name).name
                shutil.copy(file_name, target_file_name)

            logger.info("Running evaluation")
            result_path = self.eval_path / "result.txt"
            os.chdir(self.model_path)
            cmd = [
                str(self.script),
                "--mode",
                "predict_dir",
                "--bs",
                str(self.batch_size),
                "--inp_dir",
                str(self.eval_path),
                "--out_path",
                str(result_path),
                "--ckpt_path",
                str(self.ckpt_path),
            ]
            if self.use_python:
                cmd = [self.python] + cmd

            output = subprocess.check_output(cmd)
            logger.info("Evaluation finished, output: %s", output)
            file_names = [path.name for path in self.eval_path.glob("*.wav")]
            with open(result_path) as result_path:
                scores = [float(line.strip()) for line in result_path]
            score_map = dict(zip(file_names, scores))
            scores_ordered = [
                score_map[Path(file_name).name] for file_name in file_names
            ]
            return SpeechEvaluationResult(
                scores_ordered, {"utmos": scores_ordered}
            )
        finally:
            os.chdir(current_path)
            shutil.rmtree(self.eval_path)


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
