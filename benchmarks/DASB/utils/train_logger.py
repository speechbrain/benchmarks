"""Logging utilities

Authors
 * Artem Ploujnikov 2023
"""

import torch
import tarfile
import yaml
from pathlib import Path
from speechbrain.dataio.dataio import write_audio


class ArchiveTrainLogger:
    """A logger implementation that saves per-epoch progress artifacts (e.g. spectorgrams,
    audio samples, diagrams, raw tensor data, etc) in an archive

    Arguments
    ---------
    archive_path : str
        The path to the archive. It will be created if it does not exist
        and opened for appending as needed if it does

    current_path : str | path-like
        The path under which current/latest samples will be saved

    meta_path : str | path-like, optional
        The path to a simple YAML-based metadata file

    epoch_counter : speechbrain.utils.epoch_loop.EpochCounter
        The epoch counter

    epoch : int, optional
        The epoch number (not allowed if the epoch counter is supplied)
    """

    def __init__(
        self,
        archive_path,
        current_path,
        meta_path=None,
        epoch_counter=None,
        epoch=None,
    ):
        self.archive_path = Path(archive_path)
        self.current_path = Path(current_path)
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_path.mkdir(parents=True, exist_ok=True)
        self.epoch_counter = epoch_counter
        if epoch_counter and epoch is not None:
            raise ValueError(
                "An explicit epoch counter cannot be used if eppoch_counter "
                "is applied"
            )
        if epoch is None:
            epoch = 1
        self._epoch = epoch
        self.handlers = {}
        self.archive = None
        self.meta_path = Path(meta_path) if meta_path is not None else None
        for mode, handler in DEFAULT_SAVE_HANDLERS.items():
            self.add_content_handler(mode, handler)
        self._meta = {}
        self._load_meta()

    def __getitem__(self, key):
        return self._meta.get(key)

    def __setitem__(self, key, value):
        self._meta[key] = value

    def _load_meta(self):
        """Loads the metadata file"""
        if self.meta_path.exists():
            with open(self.meta_path, "r") as meta_file:
                self._meta = yaml.load(meta_file)

    def _save_meta(self):
        with open(self.meta_path, "w") as meta_file:
            yaml.dump(self.meta, meta_file)

    @property
    def epoch(self):
        """The current epoch"""
        return (
            self.epoch_counter.current
            if self.epoch_counter is not None
            else self._epoch
        )

    @epoch.setter
    def epoch(self, value):
        """The current epoch"""
        self._epoch = value

    def clear(self):
        """Clears the files currently in the current path"""
        for file_name in self.current_path.glob("*"):
            file_name.unlink()

    def __enter__(self):
        self.clear()
        if self.archive is None:
            self.archive = tarfile.open(self.archive_path, "a")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.archive is not None:
            self.archive.close()
            self.archive = None

    def save(self, name, content=None, mode="text", folder=None, **kwargs):
        """Saves the sample in the archive

        Arguments
        ---------
        name : str
            The file name
        content : object
            The content to be saved
        mode : str
            The mode identifier. The following modes are supported
            out of the box

            binary : saves arbitrary binary content
            text : saves a text file
            audio : takes an audio tensor and saves it as an
                audio file
            figure : saves a matplotlib figure as an image
        folder : None
            if specified, the file will be created in a special folder
            within the archive
        """
        current_file_name = self.current_path / name
        handler = self.handlers.get(mode)
        if handler is None:
            raise ValueError(f"Unsupported mode {mode}")
        handler(current_file_name, content, **kwargs)
        if folder is None:
            folder = Path(str(self.epoch))
        arcname = folder / Path(name)
        if self.archive is None:
            with tarfile.open(self.archive_path, "a") as archive:
                archive.add(current_file_name, str(arcname))
        else:
            self.archive.add(current_file_name, str(arcname))

    def add_content_handler(self, mode, handler):
        """Adds a content handler, which saves content
        (e.g. sound, images, reports, etc) in a file

        Arguments
        ---------
        mode : str
            The mode identifier
        handler : callable
            the save handler
        """
        self.handlers[mode] = handler


def text_handler(file_name, content):
    """A save handler for text files

    Arguments
    ---------
    file_name : str
        The file name
    content : str
        The content
    """
    with open(file_name, "w") as content_file:
        content_file.write(content)


def binary_handler(file_name, content):
    """A save handler for arbitrary binary content

    Arguments
    ---------
    file_name : str
        The file name
    content : bytes
        The content
    """
    with open(file_name, "wb") as content_file:
        content_file.write(content)


def audio_handler(file_name, content, samplerate=24000):
    """A save handler for audio content

    Arguments
    ---------
    file_name : str
        The file name
    content : torch.Tensor
        The content
    """
    write_audio(str(file_name), content, samplerate=samplerate)


def figure_handler(file_name, content):
    """A save handler for matplotlib figures

    Arguments
    ---------
    file_name : str
        The file name
    content : figure
        The matplotlib figure
    """
    content.savefig(file_name)


def tensor_handler(file_name, content):
    """A save handler for tensors, dictionaries of tensors
    or anything that can be saved via torch.save()

        Arguments
    ---------
    file_name : str
        The file name
    content : object
        Any object supported by torch.save
    """
    torch.save(content, file_name)


DEFAULT_SAVE_HANDLERS = {
    "text": text_handler,
    "binary": binary_handler,
    "audio": audio_handler,
    "figure": figure_handler,
    "tensor": tensor_handler,
}
