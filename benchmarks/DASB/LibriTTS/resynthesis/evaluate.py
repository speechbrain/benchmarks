#!/usr/bin/env/python3
"""Recipe for evaluating vocoders on resynthesis

Authors
 * Artem Ploujnikov 2024
"""


import logging
import json
import csv
import sys
import torchaudio
import speechbrain as sb

from types import SimpleNamespace
from tqdm.auto import tqdm
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.batch import PaddedData
from speechbrain.utils.distributed import run_on_main
from torch import nn


logger = logging.getLogger(__name__)


class VocoderEvaluator:
    """A standalone vocoder evaluator

    Arguments
    ---------
    hparams : dict
        Hyperparameters
    run_opts : dict
        Run options
    """
    def __init__(self, hparams, run_opts):
        self.hparams = SimpleNamespace(**hparams, run_opts=None)
        if run_opts is None:
            run_opts = {}
        self.device = run_opts.get("device", "cpu")
        self.modules = nn.ModuleDict(self.hparams.modules).to(self.device)

    def on_evaluate_start(self):
        """Invoked when evaluation starts"""
        tokenizer = (
            self.modules.tokenizer.module
            if hasattr(self.modules.tokenizer, "module")
            else self.modules.tokenizer
        )
        tokenizer.device = self.device
        if hasattr(tokenizer, "codec_vocoder"):
            tokenizer.codec_vocoder.to(self.device)
            tokenizer.codec_vocoder.device = self.device

        if self.hparams.representation_mode == "continuous":
            self.vocoder = self.hparams.vocoder(
                run_opts={"device": self.device}
            )
            if hasattr(self.vocoder, "device"):
                self.vocoder.device = self.device
            if hasattr(self.vocoder, "model"):
                self.vocoder.model.device = self.device
        self.metric = self.hparams.metric()

    def on_evaluate_end(self):
        """Invoked when evaluation ends"""
        summary = self.metric.summarize()
        output_folder = Path(self.hparams.output_folder)
        summary_file_name = output_folder / "vocoder" / "summary.json"
        summary_file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file_name, "w") as summary_file:
            json.dump(summary, summary_file, indent=4)

        mos_file_name = output_folder / "vocoder" / "mos.csv"
        with open(mos_file_name, "w") as mos_file:
            writer = csv.writer(mos_file)
            writer.writerow(["id", "score"])
            for row in zip(self.metric.ids, self.metric.scores):
                writer.writerow(row)

    def evaluate(self, dataset):
        """Evaluates the vocoder on a dataset

        Arguments
        ---------
        dataset : DynamicItemDataset
            a dataset
        """
        self.on_evaluate_start()
        dataloader = sb.dataio.dataloader.make_dataloader(dataset)
        for batch in tqdm(dataloader):
            self.evaluate_batch(batch)
        self.on_evaluate_end()

    def evaluate_batch(self, batch):
        """Evaluates a single batch

        Arguments
        ---------
        batch : PaddedBatch
            a batch"""
        batch = batch.to(self.device)
        wav_rec = self.get_wav_rec(batch)
        self.metric.append(
            ids=batch.uttid,
            wavs=wav_rec.squeeze(1),
            length=batch.sig.lengths,
            sample_rate=self.hparams.model_sample_rate
        )

    def get_wav_rec(self, batch):
        """Retrieves audio features

        Arguments
        ---------
        batch : PaddedBatch
            a batch

        Returns
        -------
        audio: torch.Tensor
            The audio representation
        """
        if self.hparams.representation_mode == "discrete":
            audio = self.modules.tokenizer.sig_to_tokens(batch.sig.data, batch.sig.lengths)
            wav_rec = self.modules.tokenizer.tokens_to_sig(audio)
        else:
            audio = self.modules.ssl_model(
                batch.sig.data,
                batch.sig.lengths,
            )
            audio = audio.permute(1, 2, 0, 3)[:, :, self.hparams.num_codebooks]
            wav_rec = self.vocoder(audio)
        return wav_rec


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
    """

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sig_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(
            sig,
            hparams["sample_rate"],
            hparams["model_sample_rate"],
        )
        return sig

    dynamic_items = [sig_pipeline]
    output_keys = ["uttid", "sig"]

    for dataset in data_info:
        dataset_dynamic_items = list(dynamic_items)
        dataset_output_keys = list(output_keys)

        dynamic_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=dataset_dynamic_items,
            output_keys=dataset_output_keys,
        )
        datasets[dataset] = dynamic_dataset

    hparams["dataloader_opts"]["shuffle"] = False
    return datasets


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        yaml = fin.read()

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
                "train_split": hparams["train_splits"],
                "valid_split": hparams["dev_splits"],
                "test_split": hparams["test_splits"],
                "save_json_train": hparams["train_json"],
                "save_json_valid": hparams["valid_json"],
                "save_json_test": hparams["test_json"],
                "sample_rate": hparams["sample_rate"],
                "skip_prep": hparams["skip_prep"],
                "max_valid_size": None,
                "skip_resample": hparams["skip_resample"],
            },
        )
    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Evaluate
    evaluator = VocoderEvaluator(hparams, run_opts)
    eval_dataset_key = hparams["eval_dataset"]
    eval_dataset = datasets[eval_dataset_key]
    logger.info("Starting evaluation on %s", eval_dataset_key)
    evaluator.evaluate(eval_dataset)
    logger.info("Evaluation ended")
