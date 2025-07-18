"""Evaluates a checkpoint using an MOS estimation tool

Authors
* Artem Ploujnikov 2024
"""

import speechbrain as sb
import json
import logging
import csv
import torch
import re
from pathlib import Path
from types import SimpleNamespace
from torch.nn import ModuleDict
from data import undo_batch
from eval import vocoder_to_device

logger = logging.getLogger(__name__)


class TokotronEvaluator:
    """An evaluator class for the TTS model

    Arguments
    ---------
    hparams: dict
        hyperparameters (as a dictionary)
    create_waveform_fn : callable
        the function that will be used to create
        waveforms (not unified across all implementations)
    device : str | torch.device
        the device
    """

    def __init__(self, hparams, create_waveform_fn, device):
        self.hparams = SimpleNamespace(**hparams)
        self.create_waveform_fn = create_waveform_fn
        self.device = device
        modules = self.hparams.modules
        self.modules = ModuleDict(modules).to(self.device)
        self.modules.model.vocoder = None
        self.enabled_evaluators = set(self.hparams.evaluations.split(","))
        evaluators = hparams.get("evaluators", {})
        if evaluators:
            self.evaluators = {
                key: evaluator_f(run_opts={"device": device})
                for key, evaluator_f in evaluators.items()
                if key in self.enabled_evaluators
            }
        else:
            self.evaluators = {}

        if not self.evaluators:
            logger.warn(
                "No evaluators were defined - this run will produce samples only"
            )

        self.attention = []
        self.compression = getattr(self.hparams, "compression", False)
        if self.compression:
            self.compression_model = self.hparams.compression_model(
                run_opts={"device": self.device}
            )
            self.modules.model.compression_model = self.compression_model

    def on_evaluate_start(self, stage, epoch):
        """Invoked when evaluation starts

        Arguments
        ---------

        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        self.stage = stage
        self.epoch = epoch
        self.output_folder = self.get_output_folder(stage, epoch)
        self.samples_folder = self.output_folder / "samples"
        self.samples_folder.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting evaluation, results will be saved in %s",
            self.output_folder,
        )
        self.create_reports()
        self.modules.model.show_inference_progress = False
        self.item_ids = []
        details_keys = list(self.evaluators.keys())
        self.details = {evaluator_key: [] for evaluator_key in details_keys}
        self.sample_text = []
        self.sample_file_names = []
        self.ref_file_names = []
        if hasattr(self.modules, "vocoder"):
            vocoder_to_device(self.modules.vocoder, self.device)

    def get_output_folder(self, stage, epoch):
        """Computes the output folder of evaluation results
        for the specified stage and epoch.

        If the folder does not exists, it will be created.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.

        Returns
        -------
        """
        output_folder = (
            Path(self.hparams.output_folder) / "eval" / stage.name.lower()
        )
        if epoch is not None:
            output_folder = output_folder / str(epoch)
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder

    def on_evaluate_end(self):
        """Runs evaluation on a dataset

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            a dataset
        """
        self.write_summary()
        logger.info("Evaluation done")

    def create_reports(self):
        """Creates report files and report writers"""
        self.report_files = {}
        self.report_writers = {}
        for evaluator_key in self.enabled_evaluators:
            columns = self.get_report_columns(evaluator_key)
            file_name = self.output_folder / f"{evaluator_key}.csv"
            report_file = open(file_name, "w")
            self.report_files[evaluator_key] = report_file
            writer = csv.DictWriter(report_file, columns)
            writer.writeheader()
            self.report_writers[evaluator_key] = writer

    def get_report_columns(self, evaluator_key):
        """Returns the columns for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            the identifier of the evaluator

        Returns
        -------
        columns : list[str]
            a list of column headers
        """
        bogus_wavs = torch.randn(2, 10000, device=self.device)
        bogus_length = torch.tensor([1.0, 1.0], device=self.device)
        if evaluator_key in self.evaluators:
            evaluator = self.evaluators[evaluator_key]
            result = evaluator.evaluate(
                wavs=bogus_wavs,
                length=bogus_length,
                text=["BOGUS"] * len(bogus_wavs),
                wavs_ref=bogus_wavs,
                length_ref=bogus_length,
            )

        return ["uttid"] + list(result.details.keys())

    def evaluate_batch(self, batch):
        """Runs evaluation on a single batch of speech

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch to be evaluated"""
        with torch.no_grad():
            batch = batch.to(self.device)
            tokens, tokens_length = batch.tokens
            infer_out = self.modules.model.infer(
                input_tokens=tokens, input_length=tokens_length
            )
            wav = self.create_waveform_fn(infer_out.audio, infer_out.length)
            self.save_samples(batch, wav, infer_out.length)
            self.item_ids.extend(batch.uttid)
            for evaluator_key, evaluator in self.evaluators.items():
                result = evaluator.evaluate(
                    wavs=wav,
                    length=infer_out.length,
                    text=batch.label_norm_eval,
                    wavs_ref=batch.sig.data,
                    length_ref=batch.sig.lengths,
                    sample_rate_ref=self.hparams.sample_rate,
                    sample_rate=self.hparams.model_sample_rate,
                )
                details = undo_batch(result.details)
                self.write_result(evaluator_key, batch.uttid, details)
                self.details[evaluator_key].extend(details)

    def write_result(self, evaluator_key, uttid, details):
        """Outputs the result details to the report for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        batch : list
            The list of IDs
        details : list
            a list of evaluation details, one dictionary per item
        """
        writer = self.report_writers[evaluator_key]
        for uttid, details_item in zip(uttid, details):
            report_details = {
                "uttid": uttid,
                **details_item,
            }
            writer.writerow(ascii_only(flatten(report_details)))
        self.report_files[evaluator_key].flush()

    def save_samples(self, batch, wav, length):
        """Saves the samples generated by the TTS system

        Arguments
        ---------
        batch : speechbrain.dataio.batch.PaddedBatch
            the batch being evaluated
        wav : torch.Tensor
            the waveform
        length: torch.Tensor
            relative lengths
        """
        wav_length_abs = (length * wav.size(1)).int()
        for item_id, infer_wav, wav_length in zip(
            batch.uttid, wav, wav_length_abs
        ):
            file_name = str(self.samples_folder / f"{item_id}_pred.wav")
            infer_wav_cut = infer_wav[: wav_length.item()].cpu()
            sb.dataio.dataio.write_audio(
                file_name,
                infer_wav_cut,
                samplerate=self.hparams.model_sample_rate,
            )
            self.sample_file_names.append(file_name)

    def write_summary(self):
        """Outputs summarized statistics"""
        summary = self.compute_summary()
        file_name = self.output_folder / "summary.json"
        with open(file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)

    def compute_summary(self):
        """Computes the summarized statistics"""
        return {
            f"{evaluator_key}_{stat_key}": value
            for evaluator_key in self.enabled_evaluators
            if evaluator_key in self.details
            for metric_key in self.hparams.eval_summary[evaluator_key][
                "descriptive"
            ]
            for stat_key, value in descriptive_statistics(
                items=self.details[evaluator_key], key=metric_key,
            ).items()
        }


def flatten(value):
    """Converts tensors to scalars and lists of strings to strings

    Arguments
    ---------
    value : dict
        the dictionary to flatten

    Returns
    -------
    result : dict
        a flattened dictionary
    """
    return {
        key: item_value.item() if torch.is_tensor(item_value) else item_value
        for key, item_value in value.items()
    }


RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def ascii_only(values):
    """Removes non-ASCII characters"""
    return {
        key: RE_NON_ASCII.sub("", value) if isinstance(value, str) else value
        for key, value in values.items()
    }


def descriptive_statistics(items, key):
    """Computes descriptive statistics for the summary

    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        """
    values = torch.tensor([item[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{key}_{stat_key}": value.item() for stat_key, value in stats.items()
    }
