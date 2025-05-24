"""Inference fit grid search for VALL-E

Curriculum inspired by Lifeiteng's VALL-E
https://github.com/lifeiteng/vall-e

Authors
 * Artem Ploujnikov 2024
"""

import speechbrain as sb
import sys
import csv
import torch
import operator
import yaml

from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from torch import nn
from tqdm.auto import tqdm
from types import SimpleNamespace
from speechbrain.dataio.dataio import clean_padding
from speechbrain.utils.logger import get_logger
from speechbrain.utils.data_utils import batch_pad_right, pad_right_to

base_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(base_dir)

from evaluation import SpeechEvaluationMetricStats  # noqa: E402
from train import undo_padding_tensor, get_offsets  # noqa: E402

logger = get_logger(__name__)

class InferenceFit:
    """An inference fit wrapper"""
    def __init__(self, hparams, run_opts):
        device = run_opts.get("device", "cpu")
        self.hparams = SimpleNamespace(**hparams)
        self.modules = nn.ModuleDict(self.hparams.modules).to(device)
        self.device = device
        self.space = self.hparams.inference_fit_space
        self.result = None
        self.evaluation_metric = SpeechEvaluationMetricStats(
            self.hparams, self.device
        )
        self.offsets = get_offsets(
            self.hparams.vocab_size, self.hparams.audio_tokens_per_step,
        )[None, None, :].to(self.device)
        if not self.hparams.use_token_offsets:
            self.offsets = torch.zeros_like(self.offsets)
        self.output_folder_rel = "eval/inference_fit"
        self.output_folder = Path(self.hparams.output_folder) / self.output_folder_rel
        self.token_model_kwargs = getattr(
            self.hparams, "token_model_kwargs", {}
        )

    def fit(self, dataset):
        """Performs infernece fitting

        Arguments
        ---------
        dataset: DynamicItemDataset
            a dataset

        Returns
        -------
        result: dict
            the fit result
        """
        self.result = []
        self.recover()
        logger.info("Parameter Space: %s", format_space(self.space))
        evaluations = self.enumerate_param_space()
        for idx, params in enumerate(tqdm(evaluations, desc="Parameter space")):
            eval_result = self.evaluate(dataset, params)
            self.result.append({"idx": idx, **params, **eval_result})
        self.best = self.find_best()
        return self.result, self.best
    
    def find_best(self):
        best = self.result[0]
        op = (
            operator.lt
            if self.hparams.inference_fit_key_metric_kind == "min"
            else operator.gt
        )
        for item in self.result[1:]:
            value = item[self.hparams.inference_fit_key_metric]
            if op(value, best[self.hparams.inference_fit_key_metric]):
                best = item
        return best

    def enumerate_param_space(self):
        return enumerate_space(self.space)

    def evaluate(self, dataset, params):
        dataloader = sb.dataio.dataloader.make_dataloader(dataset)
        params_str = format_params(params)
        logger.info("Starting evaluation of %s", params_str)
        folder_name = params_to_folder_name(params)
        self.evaluation_metric.on_evaluation_start(f"{self.output_folder_rel}/{folder_name}")
        for batch in tqdm(dataloader, desc="Evaluation run", total=len(dataset)):
            self.evaluate_batch(batch, params)
        logger.info("Finished evaluation of %s", params_str)
        self.evaluation_metric.on_evaluation_end()
        summary = self.evaluation_metric.summarize()
        metrics = {
            key: summary.get(value, 0.0)
            for key, value in self.hparams.inference_fit_metrics.items()
        }
        return metrics

    def evaluate_batch(self, batch, params):
        audio_tokens, audio_length = self.inference(batch, params)
        wav = self.create_waveform(audio_tokens, audio_length)
        wav = wav.squeeze(1)
        self.evaluation_metric.append(
            ids=batch.uttid,
            wav=wav,
            text=batch.label_norm_eval,
            length=audio_length,
            wav_ref=batch.sig.data,
            length_ref=batch.sig.lengths,
        )

    def write_report(self):
        if self.result is None:
            logger.warning("Nothing to report")
            return

        report_file_name = self.output_folder / "results.csv"
        report_file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file_name, "w") as report_file:
            columns = next(iter(self.result)).keys()
            writer = csv.DictWriter(report_file, columns)
            writer.writeheader()
            for result in self.result:
                writer.writerow(result)
        best_file_name = self.output_folder / "best.yaml"
        with open(best_file_name, "w") as best_file:
            yaml.dump(self.best, best_file)

    def inference(self, batch, params):
        """Runs TTS inference

        Arguments
        ---------
        batch : PaddedBatch
            A batch

        Returns
        -------
        audio : torch.Tensor
            A padded tensor of audio
        audio_length : torch.Tensor
            Relative lengths
        """
        prefix, prefix_length = batch.prefix
        # NOTE: ESPNET VALL-E does not support batched inference
        prefix_items = undo_padding_tensor(prefix.int(), prefix_length)
        inference = self.modules.model.inference
        inference_results = [
            inference(
                prefix=prefix_item.unsqueeze(0), opts=self._get_inference_opts(params)
            )
            for prefix_item in prefix_items
        ]
        inferred_tokens = [
            self._pad_inferred_sample(result)
            for result in inference_results
        ]
        audio, audio_length = batch_pad_right(inferred_tokens)
        audio_length = audio_length.to(self.device)
        audio = (audio - hparams["audio_token_shift"] - self.offsets).clip(0)
        return audio, audio_length

    # TODO: Duplicated in train, consider refactoring
    def _pad_inferred_sample(self, result):
        """Applies length padding to an inference result

        Arguments
        ---------
        result : list
            The VALL-E Inference output

        Returns
        -------
        sample : torch.Tensor
            A sample, padded if needed
        """
        if result[0]:
            sample = result[0][0]
        else:
            sample = torch.zeros(
                1000, self.hparams.audio_tokens_per_step, device=self.device
            )
        min_length = getattr(self.hparams, "infer_min_length", 10)
        sample_length, tracks = sample.shape
        if sample_length < min_length:
            sample = pad_right_to(
                sample,
                (min_length, tracks),
            )[0]
        return sample

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
        tokenizer = self.modules.tokenizer
        tokenizer.device = self.device
        if hasattr(tokenizer, "codec_vocoder"):
            tokenizer.codec_vocoder.to(self.device)
            tokenizer.codec_vocoder.device = self.device
        wav = tokenizer.tokens_to_sig(
            audio, **self.token_model_kwargs
        )
        wav = clean_padding(wav, length)
        wav = wav.to(self.device)
        return wav

    def _get_inference_opts(self, params):
        idx = torch.arange(self.hparams.model_vocab_size, device=self.device)[
            None, :
        ]
        tracks = torch.arange(
            self.hparams.audio_tokens_per_step, device=self.device
        )[:, None]
        if not self.hparams.use_token_offsets:
            tracks = torch.zeros_like(tracks)
        track_start = (
            self.hparams.audio_token_shift
            + tracks * self.hparams.vocab_size
        )
        if self.hparams.flip_layers:
            track_start = track_start.flip(0)
        track_end = track_start + self.hparams.vocab_size
        mask = (
            ((idx >= track_start) & (idx < track_end))
            | (idx == self.hparams.bos_index)
        ).logical_not()
        mask[
            (
                (idx >= self.hparams.special_num_tokens)
                & (idx <= self.hparams.audio_token_shift)
            ).expand_as(mask)
        ] = True
        return self.hparams.inference_opts(
            masks={self.hparams.bos_index: mask},
            **params,
            device=self.device,
        )

    def recover(self):
        test_key_kind = hparams["test_key_kind"]
        test_key = hparams["test_key"]
        kwargs = {
            f"{test_key_kind}_key": test_key
        }
        logger.info("Revovering a checkpoint")
        ckpt = self.hparams.checkpointer.recover_if_possible(**kwargs)
        if not ckpt:
            logger.error("Checkpoint not found - cannot evaluate")
            raise ValueError("No checkpoint available")
        logger.info("Checkpoint recovered: %s", ckpt)


def enumerate_space(space, entry=None, points=None):
    if points is None:
        points = []
    if not space:
        points.append(entry)
        return points
    if entry is None:
        entry = {}
    key, values = next(iter(space.items()))
    rest = dict(space)
    del rest[key]
    for value in values:
        enumerate_space(rest, {**entry, key: value}, points)
    return points


def format_space(space):
    return ", ".join(
        f"{parameter}: {values}"
        for parameter, values in space.items()
    )


def format_params(params):
    return ", ".join(
        f"{key}={value}"
        for key, value in params.items()
    )


def params_to_folder_name(params):
    params_str = "-".join(
        f"{key}-{value}"
        for key, value in params.items()
    )
    return f"eval-{params_str}"


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        yaml_content = fin.read()

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
            yaml_content = "\n".join([yaml_content, hparams_yaml])
    else:
        logger.info(
            "%s not found - not using evaluation hyperparameters",
            eval_hparams_file,
        )
    hparams = load_hyperpyyaml(yaml_content, overrides, overrides_must_match=True)
    from train import dataio_prepare, select_eval_subset # noqa
    datasets, _ = dataio_prepare(hparams)
    dataset = datasets["valid"]
    dataset = select_eval_subset(dataset, hparams)

    inference_fit = InferenceFit(hparams, run_opts)
    inference_fit.fit(dataset)
    inference_fit.write_report()
