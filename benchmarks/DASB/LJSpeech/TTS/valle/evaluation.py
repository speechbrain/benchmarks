import json
import torch
import logging
import re
import csv
from speechbrain.utils.metric_stats import MetricStats
from types import SimpleNamespace
from pathlib import Path
from utils.data import undo_batch
from torch import nn


logger = logging.getLogger(__name__)


class SpeechEvaluationMetricStats(MetricStats):
    """An aggregate metric combining multiple speech evaluators

    Arguments
    ---------
    hparams : dict | SimpleNamespace | object
        Raw hyperparameters for evaluation

    device : str
        The device on which evaluation will be performed

    """

    def __init__(self, hparams, device="cpu"):
        if isinstance(hparams, dict):
            hparams = SimpleNamespace(**hparams)
        self.hparams = hparams
        self.device = device
        modules = self.hparams.modules
        self.modules = nn.ModuleDict(modules).to(self.device)
        self.enabled_evaluators = set(self.hparams.evaluations.split(","))
        evaluators = hparams.evaluators
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

    def on_evaluation_start(self, output_folder="eval"):
        """Invoked at the beginning of the evaluation cycle.

        Arguments
        ---------
        output_folder : str | path-like
            The folder to which results will be output

        """
        logger.info("Starting evaluation")
        output_folder = Path(output_folder)
        self.output_folder = (
            output_folder
            if output_folder.is_absolute()
            else self.hparams.output_folder / output_folder
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.files = []
        details_keys = list(self.evaluators.keys())
        self.details = {evaluator_key: [] for evaluator_key in details_keys}
        self.read_reports()
        self.create_reports()
        self.item_ids = []

    def on_evaluation_end(self):
        """Invoked at the beginning of the evaluation cycle. The default
        implementation is a no-op
        """
        logger.info("Ending evaluation")
        self.write_summary()

    def create_reports(self):
        """Creates report files and report writers"""
        self.report_files = {}
        self.report_writers = {}
        for evaluator_key in self.enabled_evaluators:
            columns = self.get_report_columns(evaluator_key)
            file_name = self.output_folder / f"{evaluator_key}.csv"
            self.files.append(file_name)
            resume = file_name.exists() and file_name.stat().st_size > 0
            report_file = open(file_name, "a+")
            self.report_files[evaluator_key] = report_file
            writer = csv.DictWriter(report_file, columns)
            if not resume:
                writer.writeheader()
            self.report_writers[evaluator_key] = writer

    def read_reports(self):
        """Invoked when resuming"""
        for evaluator_key in self.enabled_evaluators:
            file_name = self.output_folder / f"{evaluator_key}.csv"
            if file_name.exists():
                logger.info("%s exists, reading")
                with open(file_name) as report_file:
                    reader = csv.DictReader(report_file)
                    for row in reader:
                        del row["uttid"]
                        row = {
                            key: handle_number(value)
                            for key, value in row.items()
                        }
                        self.details[evaluator_key].append(row)

    def get_tracker_file_name(self):
        """Determines the file name of the tracker file"""
        suffix = (
            f"_{self.hparams.eval_suffix}" if self.hparams.eval_suffix else ""
        )
        file_name = f"tracker_{self.hparams.eval_dataset}{suffix}.txt"
        return self.output_folder / file_name

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
        evaluator = self.evaluators[evaluator_key]
        result = evaluator.evaluate(
            wavs=bogus_wavs,
            length=bogus_length,
            text=["BOGUS"] * len(bogus_wavs),
            wavs_ref=bogus_wavs,
            length_ref=bogus_length,
        )

        return ["uttid"] + list(result.details.keys())

    def append(self, ids, wav, length, text, wav_ref, length_ref):
        """Appends the result of a single item

        Arguments
        ---------
        ids : str
            Utterance IDs
        wav : torch.Tensor
            Synthesized waveforms
        length : torch.Tensor
            Relative lengths of the synthesized waveforms
        text : list
            Ground truth text
        wav_ref : torch.Tensor
            Reference (ground truth) waveforms
        length_ref : torch.Tensor
            Reference lengths
        """
        with torch.no_grad():
            self.item_ids.extend(ids)
            for evaluator_key, evaluator in self.evaluators.items():
                result = evaluator.evaluate(
                    wavs=wav,
                    length=length,
                    text=text,
                    wavs_ref=wav_ref,
                    length_ref=length_ref,
                    sample_rate_ref=self.hparams.sample_rate,
                    sample_rate=self.hparams.model_sample_rate,
                )
                details = undo_batch(result.details)
                self.write_result(evaluator_key, ids, details)
                self.details[evaluator_key].extend(details)

    def write_result(self, evaluator_key, ids, details):
        """Outputs the result details to the report for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        ids : list
            The list of IDs
        details : list
            a list of evaluation details, one dictionary per item
        """
        writer = self.report_writers[evaluator_key]
        for uttid, details_item in zip(ids, details):
            report_details = {
                "uttid": uttid,
                **details_item,
            }
            writer.writerow(ascii_only(flatten(report_details)))
        self.report_files[evaluator_key].flush()

    def write_summary(self, file_name=None):
        """Outputs summarized statistics

        Arguments
        ---------
        file_name : str | path-like
            An alternative path to save the file
        """
        summary = self.summarize()
        if file_name is None:
            file_name = self.output_folder / "summary.json"
        self.files.append(file_name)
        with open(file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)

    def summarize(self, field=None):
        """Computes the summarized statistics

        Arguments
        ---------
        field : str, optional
            If specified, it will return a specific field

        Returns
        -------
        result : dict | float
            The summary - or the specified field from the sum
        """
        result = {
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
        if field is not None:
            result = result[field]
        return result

    def clear(self):
        """Deletes all the files that have been created"""
        for file_name in self.files:
            file_name.unlink()


RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def ascii_only(values):
    """Removes any non-ASCII characters from a dictionary

    Arguments
    ---------
    values : dict
        A dictionary of values

    Returns
    -------
    result : dict
        The same dictionary - but with non-ASCII strings removed"""
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
        The key of the metric for which the statistics will be computed

    Returns
    -------
    statistics : dict
        The desccriptive statistics computed
            <key>_mean : the arithmetic mean
            <key>_std : the standard deviation
            <key>_min : the minimum value
            <key>_max : the maximum value
            <key>_median : the median value
            <key>_q1 : the first quartile
            <key>_q3 : the third quartile
            <key>_iqr : the interquartile ratio
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


RE_INTEGER = re.compile(r"^-?\d+$")
RE_FLOAT = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def handle_number(value):
    """Converts a value to a number, if applicable. Strings
    that look like integers or floats will be converted to integers
    or floats.

    Arguments
    ---------
    value : str
        a string value

    Returns
    -------
    result : object
        The processed result"""
    if RE_INTEGER.match(value):
        value = int(value)
    elif RE_FLOAT.match(value):
        value = float(value)
    return value
