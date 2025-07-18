#!/usr/bin/env/python3
"""Recipe for extracting a discrete tokens with librispeech.

Authors
 * Jarod Duret 2024
"""

import os
import sys
import logging
import pathlib as pl
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech
    from libritts_prepare import prepare_libritts  # noqa

    # multi-gpu (ddp) save data preparation
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

    tokens_extractor = hparams["tokens_extractor"]
    data_folder = hparams["data_folder"]
    datasets = []
    for split in ["train", "valid", "test"]:
        json_path = hparams[f"{split}_json"]
        name = pl.Path(json_path).stem
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path, replacements={"data_root": data_folder},
        )
        datasets.append(dataset)

    merged_data = {
        key: value
        for dataset in datasets
        for key, value in dataset.data.items()
    }
    merged_dataset = DynamicItemDataset(merged_data)

    save_folder = pl.Path(hparams["save_folder"])
    logger.info("Extracting dataset tokens ...")
    tokens_extractor.extract_tokens(
        merged_dataset,
        hparams["num_codebooks"],
        (save_folder / "libritts").as_posix(),
    )

    if hparams["save_embedding"]:
        save_folder = pl.Path(hparams["save_folder"])
        logger.info("Saving embeddings ...")
        tokens_extractor.save_pretrained_embeddings(
            (save_folder / "embeddings").as_posix(),
            vocab_size=hparams["vocab_size"],
            num_codebooks=hparams["num_codebooks"],
        )
