"""
Data preparation for ASR on the Buckeye dataset.

Download: https://buckeyecorpus.osu.edu/

Author
------
Salah Zaiem, 2022
"""

import os
from tqdm import tqdm
import csv
import logging

MIN_DUR = 2
VALID_PREFIXES = ["s36", "s37"]
TEST_PREFIXES = ["s38", "s39", "s40"]
UNZIP = True
logger = logging.getLogger(__name__)


def prepare_buckeye(buckeye_dir, save_folder, unzip=True, skip_prep=False):
    """
    This function prepares the CSV for ASR training on the Buckeye dataset

    Arguments
    ---------
    buckeye_dir : str
        Path to the folder containing the Buckeye zipped folders.
    save_folder : str
        Path to the folder where the CSVs will be stored.
    unzip : bool
        If True, a script will unzip the Buckeye files.
    skip_prep: bool
        If True, data preparation is skipped.

    Returns
    -------
    None
    """
    if skip_prep:
        return

    # Check if csv files are already created
    skip_prep = any(
        os.path.isfile(os.path.join(save_folder, split + ".csv"))
        for split in ["train", "dev", "test"]
    )

    if skip_prep:
        return
    if unzip:
        unzip_buckeye(buckeye_dir)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger.info("Data_preparation...")

    TRAIN_PREFIXES = list(
        set(
            [
                x[0:3]
                for x in os.listdir(buckeye_dir)
                if x not in VALID_PREFIXES
                and x not in TEST_PREFIXES
                and x[0] == "s"
            ]
        )
    )
    logger.info("Preparing train")
    prepare_csv(buckeye_dir, save_folder, "train.csv", TRAIN_PREFIXES)
    logger.info("Preparing dev")
    prepare_csv(buckeye_dir, save_folder, "dev.csv", VALID_PREFIXES)
    logger.info("Preparing test")
    prepare_csv(buckeye_dir, save_folder, "test.csv", TEST_PREFIXES)


def treat_word_file(wrd, buckeye_dir):
    """
    This is the most technical part, the main idea here is to get rid of very short or noisy utterances.
    Baddies designate all the non verbal annotations present in the dataset, if you want to keep them
    If you are not using this script for MP3S but for broader spontaneous ASR, you may wot to change this.

    Arguments
    ---------
    wrd: str
        Path to the .words file in the Buckeye directory.
    buckeye_dir : str
        Path to the folder containing the Buckeye unzipped files.
    Returns
    -------
    csvs: list
        The list of csv lines for that .wrd file.

    """
    wavname = wrd.split("/")[-1].split(".")[0]
    with open(wrd, "r") as word_file:
        lines = word_file.read().splitlines()
        considered_lines = lines[9:]
        words = []
        baddies = []
        # This first inspection will not consider sequences containing "baddies"
        for line in considered_lines:
            if len(line.split()) < 3:
                continue
            start, _, word = line.split()[0:3]
            words.append(word)
            if "<" in word:
                baddies.append(word[:-1])
        baddies = set(baddies)
        baddies.add("uh")
        sentences = []
        sentences_starts = []
        sentences_ends = []
        sentence = []
        duration = 0
        durations = []
        starts = []
        words = []
        current_start = 0
        for ind, line in enumerate(considered_lines):
            if len(line.split()) < 3:
                continue
            start, _, word = line.split()[0:3]
            starts.append(float(start))
            words.append(word[:-1])
        # A second pass now removes the utterances that are shorter than MIN_DUR
        for ind, word in enumerate(words):

            start = starts[ind]
            if word in baddies:
                if duration != 0:
                    duration += start - starts[ind - 1]
                if duration > MIN_DUR:
                    if len(sentence) > 2:
                        sentences.append(" ".join(sentence))
                        sentences_ends.append(starts[ind - 1])
                        sentences_starts.append(current_start)
                        durations.append(starts[ind - 1] - current_start)

                    sentence = []
                    duration = 0
                    current_start = 0
                else:
                    sentence = []
                    duration = 0
                    current_start = 0

            else:
                if current_start == 0:
                    current_start = starts[ind - 1]

                sentence.append(word)
                if duration != 0:
                    duration += start - starts[ind - 1]
                if duration == 0:
                    duration = -1
                if duration == -1:
                    duration = start - starts[ind - 1]

        csvs = []
        # Finally, now the sentences are selected, prepare the csv lines
        for ind, sentence in enumerate(sentences):
            csv_line = [
                wavname + "_" + str(ind),
                str(durations[ind]),
                os.path.join(buckeye_dir, wavname + ".wav"),
                wavname[0:3],
                sentences_starts[ind],
                sentences_ends[ind],
                sentence.upper(),
            ]
            csvs.append(csv_line)
        return csvs


def unzip_buckeye(buckeye_dir):
    """
    Little script to unzip the Buckeye dataset files

    Arguments
    ---------
    buckeye_dir : str
        Path to the folder containing the Buckeye zipped folders.

    Returns
    -------
    None
    """
    files = os.listdir(buckeye_dir)
    for zip_fil in files:
        if ".py" not in zip_fil:
            os.system(
                f"unzip -q {os.path.join(buckeye_dir,zip_fil)} -d {buckeye_dir}"
            )
    files = os.listdir(buckeye_dir)
    for under_dir in files:
        if os.path.isdir(os.path.join(buckeye_dir, under_dir)):
            under_dir_full_path = os.path.join(buckeye_dir, under_dir)
            zips = os.listdir(under_dir_full_path)
            for z in zips:
                if z != under_dir.split(".")[0]:
                    os.system(
                        f"unzip -q {os.path.join(under_dir_full_path, z)} -d {buckeye_dir}"
                    )


def prepare_csv(buckeye_dir, save_folder, csv_file, prefixes):
    """
    This function prepares and outputs the csv for a given split defined by its recording prefixes.

    Arguments
    ---------
    buckeye_dir : str
        Path to the folder containing the Buckeye unzipped files.
    save_folder: str
        Path to the folder where the preparation is saved.
    csv_file: str
        Name of the file that will be saved (corresponding to the split)
    prefixes: list
        List of prefixes of recordings defining the elements that go into this split

    Returns
    -------
    None
    """
    csv_lines = [
        ["ID", "duration", "wav", "spk_id", "start_seg", "end_seg", "wrd"]
    ]
    split_elements = [
        x
        for x in os.listdir(buckeye_dir)
        if x[0:3] in prefixes and ".words" in x
    ]
    for wrd_name in tqdm(split_elements):
        wrd = os.path.join(buckeye_dir, wrd_name)
        csv_from_wrd = treat_word_file(wrd, buckeye_dir)
        csv_lines += csv_from_wrd
    with open(os.path.join(save_folder, csv_file), mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)
