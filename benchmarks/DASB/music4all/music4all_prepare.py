"""
Music4All data preparation.
Download: https://sites.google.com/view/contact4music4all

Authors
 * Pooneh Mousavi 2024
"""

import os
import csv
import json
import random
import logging
from types import SimpleNamespace
from tqdm import tqdm
import os
import csv
import random
import json
from speechbrain.dataio.dataio import (
    read_audio_info,
)


logger = logging.getLogger(__name__)
METADATA_CSV = "id_metadata.csv"
WAVS = "audios"
DURATIONS = "durations"
FROZEN_SPLIT="frozen_split.json"

logger = logging.getLogger(__name__)


def prepare_music4all(
    data_folder,
    save_folder,
    splits=["train", "valid","test"],
    split_ratio=[80, 10, 10],
    seed=1234,
    skip_prep=False,
    frozen_split_path=None,
    device="cpu",
):
    """
    Prepares the csv files for the Music4All datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored
    save_folder : str
        The directory where to store the csv/json files
    splits : list
        List of dataset splits to prepare
    split_ratio : list
        Proportion for dataset splits
    seed : int
        Random seed
    skip_prep : bool
        If True, skip preparation
    frozen_split_path : str | path-like
        The path to the frozen split file (used to standardize multiple
        experiments)
    device : str
        Device for to be used for computation (used as required)

    Returns
    -------
    None

    Example
    -------
    >>> data_folder = 'data/music4all/'
    >>> save_folder = 'save/'
    >>> splits = ['train', 'valid','test']
    >>> split_ratio = [80, 10, 10]
    >>> seed = 1234
    >>> prepare_music4all(data_folder, save_folder, splits, split_ratio, seed)
    """
    # Sets seeds for reproducible code
    random.seed(seed)

    if skip_prep:
        return
    
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder):
        logger.info("Skipping preparation, completed in previous run.")
        return


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    meta_csv = os.path.join(data_folder, METADATA_CSV)
    wavs_folder = os.path.join(data_folder, WAVS)
    if frozen_split_path is  None:
        frozen_split_path = os.path.join(save_folder, FROZEN_SPLIT)


    # Additional check to make sure metadata.csv and wavs folder exists
    assert os.path.exists(meta_csv), "metadata.csv does not exist"
    assert os.path.exists(wavs_folder), "wavs/ folder does not exist"

    # Prepare data splits
    msg = "Creating csv file for music4all Dataset.."
    logger.info(msg)
    # Get the splits
    splits_data = split_sets(meta_csv, splits, split_ratio, frozen_split_path)

    # Dynamically create CSV files for each split
    for split in splits_data:
        logger.info(f"Start processing {split} data.")
        save_json_path = os.path.join(save_folder, f"{split}.json")  # Dynamic filename
        create_csv(splits_data[split], wavs_folder, save_json_path)
        logger.info(f"Saved {split} data to {save_json_path}")



def skip(splits, save_folder):
    """
    Detects if the ljspeech data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True


    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, f"{split}.json")):
            skip = False
    return skip



# Function to split IDs based on ratios
def split_sets(meta_file, splits, split_ratio, frozen_split_path):
    """
    Splits data into train, valid, and test sets based on given ratios. 
    Checks if frozen splits already exist and uses them if available.

    Parameters:
        data_folder (str): Path to the folder containing `meta.csv`.
        splits (list): List of split names, e.g., ["train", "valid", "test"].
        split_ratio (list): Ratios for train, valid, and test splits.
        frozen_split_path (str): Path to save/load the frozen splits JSON file.

    Returns:
        dict: A dictionary with keys as split names containing the split IDs.
    """
    # Check if frozen splits already exist
    if os.path.exists(frozen_split_path):
        logger.info(f"Loading frozen splits from {frozen_split_path}")
        with open(frozen_split_path, 'r') as f:
            splits_data = json.load(f)
        return splits_data

    with open(meta_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')  # Using '\t' as delimiter for tab-separated file
        # Extract all IDs
        all_ids = [row['id'] for row in reader]

    # Shuffle the IDs and calculate split indices
    random.shuffle(all_ids)
    total = len(all_ids)
    train_end = int(total * (split_ratio[0] / 100))
    valid_end = train_end + int(total * (split_ratio[1] / 100))

    # Generate new splits
    splits_data = {
        splits[0]: all_ids[:train_end],         # Train
        splits[1]: all_ids[train_end:valid_end],  # Valid
        splits[2]: all_ids[valid_end:]         # Test
    }

    # Save the splits to frozen_split_path
    logger.info(f"Saving splits to {frozen_split_path}")
    with open(frozen_split_path, 'w') as f:
        json.dump(splits_data, f, indent=4)

    return splits_data

# Function to create CSV files
def create_csv(ids, audio_folder, output_json):
    """
    Creates a CSV file with `id`, `audio_path`, and `duration` for the given IDs.

    Parameters:
        ids (list): List of IDs to include in the CSV.
        audio_folder (str): Folder containing the audio files.
        output_csv (str): Path to the output CSV file.
    """
    json_dict = {}
    for file_id in tqdm(ids, desc="Processing split", unit="file_id"):
        audio_path = os.path.join(audio_folder, f"{file_id}.mp3")
        if os.path.exists(audio_path):
            try:
                # Get the duration of the audio file
                info = read_audio_info(audio_path)
                duration = info.num_frames / info.sample_rate
                json_dict[file_id] = {
                    "uttid": file_id,
                    "wav": audio_path,
                    "duration": duration
                }
            except Exception as e:
                logger.warn(f"Error processing file {audio_path}")
                continue
    
    # Writing the dictionary to the json file
    with open(output_json, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{output_json} successfully created!")

# # Example Usage
# if __name__ == "__main__":
#     data_folder = '/home/ubuntu/music4all'
#     save_folder = 'save/'
#     splits = ["train", "valid", "test"]
#     split_ratio = [1, 10, 80]  # Train, Valid, Test percentages
#     seed = 1234
#     prepare_music4all(data_folder, save_folder, splits, split_ratio, seed)



