"""WSJ0Mix data preparation.

Authors
 * Luca Della Libera 2023
"""

import csv
import logging
import os
from typing import Optional

import torchaudio


__all__ = ["prepare_wsj0mix"]


# Logging configuration
logging.basicConfig(
    level=logging.INFO,  # format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


def prepare_wsj0mix(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    num_speakers: "int" = 2,
    version: "str" = "wav8k/min",
) -> "None":
    """Prepare data manifest CSV files for the WSJ0Mix dataset
    (see https://catalog.ldc.upenn.edu/LDC93s6a and https://www.dropbox.com/s/gg524noqvfm1t7e/create_mixtures_wsj023mix.zip?dl=1).

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    num_speakers:
        The number of speakers (2 or 3).
    version:
        The dataset version.

    Raises
    ------
    ValueError
        If an invalid argument value is given.

    Examples
    --------
    >>> # Expected folder structure: wsj0-2mix-8k-min/wsj0-mix/2speakers/wav8k/min
    >>> prepare_wsj0mix("wsj0-2mix-8k-min/wsj0-mix", num_speakers=2, version="wav8k/min")

    """
    if not save_folder:
        save_folder = data_folder
    if num_speakers not in [2, 3]:
        raise ValueError(
            f"`num_speakers` ({num_speakers}) must be either 2 or 3"
        )
    version = version.replace("/", os.sep)

    # Write output CSV for each split
    for split in ["tr", "cv", "tt"]:
        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Split: {split}")

        split_folder = os.path.join(f"{num_speakers}speakers", version, split)

        # Mix files
        mix_folder = os.path.join(split_folder, "mix")
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

        headers = ["ID", "duration", "mix_wav"] + [
            f"src{i}_wav" for i in range(1, num_speakers + 1)
        ]
        output_csv = os.path.join(
            save_folder, f"wsj0_{num_speakers}mix_{split}.csv"
        )
        _LOGGER.info(f"Writing {output_csv}...")
        with open(output_csv, "w", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=headers)
            csv_writer.writeheader()
            for i in range(len(mix_wavs)):
                ID = f"wsj0_{num_speakers}mix_{split}_{str(i).zfill(5)}"
                mix_wav = mix_wavs[i]
                src_wavs = [src_wavs[i] for src_wavs in all_src_wavs]
                info = torchaudio.info(
                    mix_wav.replace("$DATA_ROOT", data_folder)
                )
                duration = info.num_frames / info.sample_rate
                entry = dict(zip(headers, [ID, duration, mix_wav, *src_wavs]))
                csv_writer.writerow(entry)

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info("Done!")
