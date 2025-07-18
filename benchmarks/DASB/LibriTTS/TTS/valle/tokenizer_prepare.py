"""A script to prepare annotations for tokenizers

"""

import json
import re

from pathlib import Path
from speechbrain.lobes.models.g2p.dataio import build_token_char_map
from speechbrain.utils.logger import get_logger


logger = get_logger(__name__)
MULTI_SPACE = re.compile(r"\s{2,}")


def phn2txt(phn, phoneme_map):
    """Encodes phonemes using a character map for use with SentencePiece

    Arguments
    ---------
    phn: list
        a list of original phonemes (ARPABET)
    phoneme_map: dict
        the phoneme-to-character map

    Returns
    -------
    value: str
        the mapped string representation
    """
    value = "".join(phoneme_map[phoneme] for phoneme in phn).strip()
    value = MULTI_SPACE.sub(" ", value)
    return value


def prepare_annotation(src, destination_file_name, phonemes):
    """Prepares the annotation file

    Arguments
    ---------
    src: datasets.arrow_dataset.Dataset
        the source dataset
    destination_file_name: str
        the path to the annotation file to be created
    phonemes: list
        the list of phonemes
    """
    phoneme_map = build_token_char_map(phonemes)
    annotation = {
        key: {
            "label": item["label"],
            "phonemes": phn2txt(item["phn"], phoneme_map),
        }
        for key, item in src.items()
    }
    with open(destination_file_name, "w", encoding="utf-8") as dst_file:
        json.dump(annotation, dst_file, indent=2)


DATA_SPLITS = ["train", "valid", "test"]


def prepare_tokenizer(splits, save_folder, input, phonemes):
    """Prepares annotations for the tokenizer

    Arguments
    ---------
    datasets: list
        the list of dataset splits
    save_folder: str
        the path to the folder where annotations will be saved
    input : str
        identifies what type of input will be used (text or phonemes)
    phonemes: list
        the list of phonemes
    """
    save_folder = Path(save_folder)
    if input == "text":
        for key in splits:
            src_file_name = save_folder / f"{key}.json"
            destination_file_name = (
                save_folder / f"tokenizer_annotation_{key}.json"
            )
            destination_file_name.symlink_to(src_file_name)
    else:
        for key in splits:
            destination_file_name = (
                save_folder / f"tokenizer_annotation_{key}.json"
            )
            if destination_file_name.exists():
                logger.info(
                    "Annotation file '%s' already exists", destination_file_name
                )
            else:
                logger.info(
                    "Creating tokenizer annotation '%s'", destination_file_name,
                )
                data_file_name = save_folder / f"{key}.json"
                with open(data_file_name) as data_file:
                    data = json.load(data_file)
                prepare_annotation(
                    src=data,
                    destination_file_name=destination_file_name,
                    phonemes=phonemes,
                )
