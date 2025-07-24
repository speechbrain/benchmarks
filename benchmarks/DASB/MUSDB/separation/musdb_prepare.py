import csv
import logging
import os
from typing import Optional, Sequence

from tqdm import tqdm

import speechbrain as sb


__all__ = ["prepare_musdb"]

SOURCE_NAMES = [
    "bass.wav",
    "drums.wav",
    "other.wav",
    "vocals.wav",
]

# Workaround to use fastest backend (SoundFile)
try:
    import torchaudio

    torchaudio._backend.utils.get_available_backends().pop("ffmpeg", None)
except Exception:
    pass

# Logging configuration
logging.basicConfig(
    level=logging.INFO,  # format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


def prepare_musdb(
    data_folder: "str",
    save_folder: "Optional[str]" = None,
    splits: "Sequence[str]" = ("train", "eval", "validation"),
) -> "None":
    """Prepare data manifest CSV files for the MUSDB dataset

    Arguments
    ---------
    data_folder:
        The path to the dataset folder.
    save_folder:
        The path to the folder where the data manifest CSV files will be stored.
        Default to `data_folder`.
    splits:
        The dataset splits to prepare.
    num_sources:
        The number of speakers (1, 2 or 3).

    Raises
    ------
    ValueError
        If an invalid argument value is given.
    RuntimeError
        If one of the expected split folders is missing.

    Examples
    --------
    >>> # Expected folder structure: MUSDB/{train, test}/<track_name>/{mixture.wav, bass.wav, other.wav, drums.wav, vocals.wa}
    >>> prepare_musdb("MUSDB", num_sources=4)

    """
    if not save_folder:
        save_folder = data_folder

    train_data = []
    test_data = []
    valid_data = []

    # Iterate over train and test splits
    for split in splits:
        split_dir = os.path.join(data_folder, split)

        # Check if the split directory exists
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Skipping.")
            continue

        # Walk through the subdirectories of the split (tracks)
        tracks = os.listdir(split_dir)
        for i, track_id in enumerate(tqdm(tracks, desc=split)):
            track_dir = os.path.join(split_dir, track_id)
            # Ensure the track directory exists and contains the required files
            required_files = [
                "mixture.wav",
                "bass.wav",
                "drums.wav",
                "other.wav",
                "vocals.wav",
            ]
            file_paths = {}

            for file_name in required_files:
                file_path = os.path.join(track_dir, file_name)
                if os.path.exists(file_path):
                    file_paths[file_name] = file_path
                else:
                    print(
                        f"Warning: {file_name} missing in {track_dir}. Skipping track."
                    )
                    file_paths = None
                    break  # If any file is missing, skip the current track

            # If all required files are found, process the track
            if file_paths:
                # Get the duration of the 'mixture.wav' file
                mixture_wav_path = file_paths["mixture.wav"]
                info = sb.dataio.dataio.read_audio_info(mixture_wav_path)
                duration = info.num_frames / info.sample_rate

                # Prepare the row for the CSV
                row = [
                    split,
                    track_id,  # ID
                    duration,  # duration
                    file_paths["mixture.wav"],  # mixture_wav
                    file_paths["bass.wav"],
                    file_paths["drums.wav"],
                    file_paths["other.wav"],
                    file_paths["vocals.wav"],
                ]

                # Add the row to the appropriate data list
                if split == "train":
                    train_data.append(row)
                elif split == "eval":
                    test_data.append(row)
                elif split == "validation":
                    valid_data.append(row)

    # Define the CSV file headers
    headers = [
        "split",
        "ID",
        "duration",
        "mixture_wav",
        "bass_wav",
        "drums_wav",
        "other_wav",
        "vocals_wav",
    ]

    # Write the CSV files for each split
    for data, split in [
        (train_data, "train"),
        (test_data, "eval"),
        (valid_data, "validation"),
    ]:
        output_csv = os.path.join(save_folder, f"{split}.csv")

        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)
        print(f"CSV file created for {split}: {output_csv}")

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info("Done!")
