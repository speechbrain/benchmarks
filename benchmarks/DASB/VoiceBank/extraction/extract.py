#!/usr/bin/env/python3

"""Recipe for extracting a discrete tokens with VoiceBank.

Authors
 * Jarod Duret 2024
 * Luca Della Libera 2024
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

print(base_dir)

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

    # Dataset prep (parsing voicebank)
    from voicebank_prepare import prepare_voicebank  # noqa

    # multi-gpu (ddp) save data preparation
    os.makedirs(hparams["save_folder"], exist_ok=True)
    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "num_valid_speakers": hparams["num_valid_speakers"],
        },
    )

    tokens_extractor_in = hparams["tokens_extractor_in"]
    tokens_extractor_out = hparams["tokens_extractor_out"]
    data_folder = hparams["data_folder"]

    datasets = []
    for csv_path in [hparams["train_csv"], hparams["valid_csv"], hparams["test_csv"]]:
        name = pl.Path(csv_path).stem
        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path, replacements={"DATA_ROOT": data_folder},
        )
        datasets.append(dataset)

    merged_data = {
        key: value
        for dataset in datasets
        for key, value in dataset.data.items()
    }
    merged_dataset = DynamicItemDataset(merged_data)

    save_folder = pl.Path(hparams["save_folder"])
    logger.info("Extracting dataset input tokens ...")
    tokens_extractor_in.extract_tokens(
        merged_dataset,
        hparams["num_codebooks"],
        (save_folder / "input").as_posix(),
    )

    if hparams["save_embedding"]:
        save_folder = pl.Path(hparams["save_folder"])
        logger.info(f"Saving embeddings ...")
        tokens_extractor_in.save_pretrained_embeddings(
            (save_folder / "embeddings" / "input").as_posix(),
            vocab_size=hparams["vocab_size"],
            num_codebooks=hparams["num_codebooks"],
        )

    logger.info("Extracting dataset output tokens ...")
    tokens_extractor_out.extract_tokens(
        merged_dataset,
        hparams["num_codebooks"],
        (save_folder / "output").as_posix(),
    )

    if hparams["save_embedding"]:
        save_folder = pl.Path(hparams["save_folder"])
        logger.info(f"Saving embeddings ...")
        tokens_extractor_out.save_pretrained_embeddings(
            (save_folder / "embeddings" / "output").as_posix(),
            vocab_size=hparams["vocab_size"],
            num_codebooks=hparams["num_codebooks"],
        )
