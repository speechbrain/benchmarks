"""LibriMix data preparation.

Authors
 * Luca Della Libera 2023
"""

import csv
import logging
import os
from typing import Optional

import torchaudio


__all__ = ["prepare_librimix"]


# Logging configuration
logging.basicConfig(
    level=logging.INFO,  # format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


def prepare_librimix(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    num_speakers: "int" = 2,
    add_noise: "bool" = False,
    version: "str" = "wav8k/min",
) -> "None":
    """Prepare data manifest CSV files for the LibriMix dataset
    (see https://github.com/JorisCos/LibriMix).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    num_speakers:
        The number of speakers (2 or 3).
    add_noise:
        True to add WHAM noise, False otherwise.
    version:
        The dataset version.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> # Expected folder structure: LibriMix/Libri2Mix/wav8k/min
    >>> prepare_librimix("LibriMix", num_speakers=2, version="wav8k/min")

    """
    if not save_folder:
        save_folder = data_folder
    if num_speakers not in [2, 3]:
        raise ValueError(
            f"`num_speakers` ({num_speakers}) must be either 2 or 3"
        )
    version = version.replace("/", os.sep)

    # Write output CSV for each split
    for split in ["train-360", "dev", "test"]:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        split_folder = os.path.join(f"Libri{num_speakers}Mix", version, split)

        # Mix files
        mix_folder = os.path.join(
            split_folder, "mix_both" if add_noise else "mix_clean"
        )
        mix_wavs = os.listdir(os.path.join(data_folder, mix_folder))
        mix_wavs = [
            os.path.join("$DATA_ROOT", mix_folder, mix_wav)
            for mix_wav in mix_wavs
        ]

        # Original files
        all_src_wavs = []
        for i in range(1, num_speakers + 1):
            src_folder = os.path.join(split_folder, f"s{i}")
            src_wavs = os.listdir(os.path.join(data_folder, src_folder))
            src_wavs = [
                os.path.join("$DATA_ROOT", src_folder, x) for x in src_wavs
            ]
            all_src_wavs.append(src_wavs)

        assert all(len(x) == len(mix_wavs) for x in all_src_wavs)

        if add_noise:
            # Noise files
            noise_folder = os.path.join(split_folder, "noise")
            noise_wavs = os.listdir(os.path.join(data_folder, noise_folder))
            noise_wavs = [
                os.path.join("$DATA_ROOT", noise_folder, noise_wav)
                for noise_wav in noise_wavs
            ]

            assert len(noise_wavs) == len(mix_wavs)

        headers = (
            ["ID", "duration", "mix_wav"]
            + [f"src{i}_wav" for i in range(1, num_speakers + 1)]
            + (["noise_wav"] if add_noise else [])
        )
        output_csv = os.path.join(
            save_folder, f"librimix_{num_speakers}mix_{split}.csv"
        )
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i in range(len(mix_wavs)):
                ID = f"librimix_{num_speakers}mix_{split}_{str(i).zfill(5)}"
                mix_wav = mix_wavs[i]
                src_wavs = [src_wavs[i] for src_wavs in all_src_wavs]
                info = torchaudio.info(
                    mix_wav.replace("$DATA_ROOT", data_folder)
                )
                duration = info.num_frames / info.sample_rate
                entry = dict(
                    zip(
                        headers,
                        [ID, duration, mix_wav, *src_wavs]
                        + ([noise_wavs[i]] if add_noise else []),
                    )
                )
                csv_writer.writerow(entry)

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info("Done!")
