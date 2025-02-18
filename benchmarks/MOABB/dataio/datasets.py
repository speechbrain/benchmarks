"""PyTorch Dataset implementations for raw and epoched EEG data.

Author
------
Drew Wagner, 2025
Bruno Aristimunha, 2025
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Hashable, Iterable, Optional, Self, TypedDict

import mne
import numpy as np
from mne_bids import BIDSPath, get_bids_path_from_fname, read_raw_bids
from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset as BaseMOABBDataset
from moabb.datasets.bids_interface import camel_to_kebab_case

from torch.utils.data import Dataset

from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import provides, takes



class RawEEGSample(TypedDict, total=False):
    """Default dictionary keys provided by `~RawEEGDataset`.

    NOTE: The actual keys available will depend on the dataset's
    `dynamic_items` and `output_keys`.
    """

    id: Hashable
    fpath: str  # Filepath to RAW data
    raw: mne.io.RawArray
    info: mne.Info
    # BIDS entities
    subject: Optional[str]
    session: Optional[str]
    task: Optional[str]
    acquisition: Optional[str]
    run: Optional[str]
    processing: Optional[str]
    space: Optional[str]
    recording: Optional[str]
    split: Optional[str]
    description: Optional[str]


class EpochedEEGSample(RawEEGSample):
    """Default dictionary keys provided by `~EpochedEEGDataset`.

    NOTE: The actual keys available will depend on the dataset's
    `dynamic_items` and `output_keys`.
    """

    onset: int
    epoch: np.ndarray


class RawEEGDataset(DynamicItemDataset):
    """Dataset which loads raw data from a BIDS directory.

    By default, data is loaded lazily from disk, but can optionally be preloaded to memory.

    Supports additional dynamic transformations. See Speechbrain's `~DynamicItemDataset` for
    more details.

    NOTE: This class provides access to the raw EEG data. To access the epoched data,
    use `~EpochedEEGDataset`.

    Arguments
    ---------
    data : dict[Any, dict]
        A dictionary which maps unique sample identifiers to sample metadata.
    preload : bool, optional
        Whether or not to preload the raw data into memory. If False, then data will be
        lazily loaded from disk. Defaults to False.
    verbose : bool, optional
        Whether or not to enable verbose logging for MNE / MOABB operations. If None, then
        default logging levels will be used. Defaults to None.
    dynamic_items : list, optional
        See `~DynamicItemDataset`. Note that a dynamic item "raw" is automatically include which
        reads the data file defined by "fpath".
    output_keys : dict, list, optional
        See `~DynamicItemDataset`.
    """

    suffix = "eeg"  # By default, only EEG data will be included

    def __init__(
        self,
        data,
        preload=False,
        verbose=None,
        dynamic_items=(),
        output_keys=(),
    ):
        self.verbose = verbose
        dynamic_items = [self._make_load_raw_dynamic_item(preload)] + list(
            dynamic_items
        )
        super().__init__(
            data, dynamic_items=dynamic_items, output_keys=output_keys
        )

        if preload:
            # Iterate once through the dataset to warmup the cache and preload data
            with self.output_keys_as(["raw"]):
                for _ in self:
                    pass

    @classmethod
    def from_bids(
        cls,
        bids_path: BIDSPath | Path | str,
        json_path: Path | str,
        subjects=None,
        **cls_kwargs,
    ) -> Self:
        """Creates a DynamicItemDataset from a BIDS EEG Dataset.

        Arguments
        ---------
        bids_path : BIDSPath, Path, str
            Path to the BIDS directory which should be read.
        json_path : Path, str
            The path to save or load the JSON index.
        subjects : list[str], optional
            Optionally process only a subset of subjects. Defaults to all subjects.
        **cls_kwargs
            Additional arguments to pass to the `~RawEEGDataset.__init__` function.

        Returns
        -------
            DynamicItemDataset initialized to read the BIDS dataset.
        """
        if not isinstance(bids_path, BIDSPath):
            bids_path = BIDSPath(root=bids_path)
        json_data = cls.load_or_create_json_data_from_bids(
            bids_path, json_path, subjects=subjects
        )

        return cls(data=json_data, **cls_kwargs)  # type: ignore

    @classmethod
    def from_moabb(
        cls,
        dataset: BaseMOABBDataset,
        json_path: str | Path,
        subjects=None,
        save_path: Optional[str] = None,
        **cls_kwargs,
    ) -> Self:
        """Creates a DynamicItemDataset from a MOABB Dataset.

        The MOABB dataset will be first converted to BIDS format, and
        saved on disk using MOABB's raw caching mechanism.

        NOTE: Two copies of the dataset will be stored on disk, the original
        data downloaded by MOABB, and the processed data normalized in BIDS
        format.

        Arguments
        ---------
        dataset : moabb.datasets.BaseDataset
            The MOABB dataset instance to convert to BIDS and index.
        json_path : str, Path
            The path to save or load the JSON index.
        subjects : list[str], optional
            Optionally process only a subset of subjects. Defaults to all subjects.
        save_path : str, optional
            Optional path where the converted BIDS dataset should be saved. Defaults to default
            MNE data directory.
        **cls_kwargs
            Additional arguments to pass to the `~RawEEGDataset.__init__` function.

        Returns
        -------
        RawEEGDataset
            DynamicItemDataset initialized to read the MOABB dataset.
        """
        # Reading the mne-python.json
        json_path = Path(json_path)
        if json_path.exists():
            with json_path.open() as fp:
                json_data = json.load(fp)

            return cls(data=json_data, **cls_kwargs)

        mne_path = Path(dl.get_dataset_path(dataset.code, save_path))

        cache_dir = f"MNE-BIDS-{camel_to_kebab_case(dataset.code)}"
        cache_path = mne_path / cache_dir

        subject_list = (
            subjects if subjects is not None else dataset.subject_list
        )
        dataset.download(subject_list) # ??

        # Convert from MOABB format to BIDS
        for sub in subject_list:
            dataset.get_data(
                subjects=[sub],
                cache_config=dict(use=True, save_raw=True, path=mne_path),
            )

        @provides("dataset")
        def _add_dataset():
            return dataset

        # Dynamically add the dataset reference
        dynamic_items = list(cls_kwargs.pop("dynamic_items", []))
        dynamic_items.insert(0, _add_dataset)

        return cls.from_bids(
            bids_path=cache_path,
            json_path=json_path,
            subjects=subjects,
            dynamic_items=dynamic_items,
            **cls_kwargs,
        )

    def __getitem__(self, index) -> RawEEGSample:
        return super().__getitem__(index)  # type: ignore

    @classmethod
    def load_or_create_json_data_from_bids(
        cls,
        bids_path: BIDSPath,
        json_path: Path | str,
        subjects=None,
    ) -> dict[str, dict]:
        """Indexes the BIDS directory and saves the result to a JSON file, or loads
        the index from JSON if it already exists.

        Arguments
        ---------
        bids_path : BIDSPath
            The BIDS root directory to index
        json_path : Path, str
            The .json path where the index will be cached
        subjects : list[str], optional
            Index only a subset of subjects. Defaults to all subjects.

        Returns
        -------
        dict[str, dict]
            Returns a mapping from unique sample ID to sample metadata
        """
        json_path = Path(json_path)
        if json_path.exists():
            with json_path.open() as fp:
                json_data = json.load(fp)
        else:
            json_data = cls.json_data_from_bids_path(bids_path)

            with json_path.open("w") as fp:
                json.dump(json_data, fp)

        if subjects is not None:
            json_data = {
                uid: data
                for uid, data in json_data.items()
                if data.get("subject") in subjects
            }

        return json_data

    @classmethod
    def json_data_from_bids_path(cls, bids_path: BIDSPath) -> dict[str, dict]:
        """Indexes all BIDSPaths which match the desired suffix, and returns as a dict.

        Arguments
        ---------
        bids_path : BIDSPath
            The BIDS root directory to index.

        Returns
        -------
        dict[str, dict]
            Returns a mapping from unique sample ID to sample metadata
        """
        json_data: dict[str, dict] = {}

        matched_paths: Iterable[BIDSPath] = bids_path.update(
            suffix=cls.suffix
        ).match(ignore_json=True)
        for path in matched_paths:
            uid = path.fpath.name
            json_data[uid] = path.entities
            json_data[uid]["fpath"] = str(path.fpath)
        return json_data

    def _make_load_raw_dynamic_item(self, preload: bool):
        @takes("fpath")
        @provides("info", "raw")
        def _load_raw(fpath: str):
            raw = self._read_raw_bids_cached(fpath, preload)

            yield raw.info
            yield raw

        return _load_raw

    @cache
    def _read_raw_bids_cached(
        self, fpath: str, preload: bool
    ) -> mne.io.RawArray:
        bids_path = get_bids_path_from_fname(fpath)

        return read_raw_bids(
            bids_path=bids_path,
            extra_params=dict(preload=preload),
            verbose=self.verbose,
        )


class EpochedEEGDataset(RawEEGDataset):
    """Dataset which loads pre-computed epochs from a BIDS directory.

    By default data is loaded lazily from disk, but can optionally be preloaded to memory.

    Supports additional dynamic transformations. See Speechbrain's `~DynamicItemDataset` for
    more details.

    NOTE: This class provides access to the epoched EEG data. To access only the raw data,
    use `~RawEEGDataset`.

    Arguments
    ---------
    data : dict[Any, dict]
        A dictionary which maps unique sample identifiers to sample metadata.
    tmin : float, optional
        Crops the data using this offset in seconds from the beginning of the epoch. Defaults to 0.
    tmax : float, optional
        Crops the data using this offset in seconds from the begininng of the epoch. If None, then
        the data will not be cropped to an upper bound. Defaults to None.
    dynamic_items : list, optional
        See `~DynamicItemDataset`. Note that a dynamic item "epoch" is automatically include which
        reads the section of the data file defined by "fpath" and "onset".

        NOTE: These dynamic items will be applied after epoching.
    output_keys : dict, list, optional
        See `~DynamicItemDataset`.
    **kwargs
        Additional keyword arguments which will be passed to `~RawEEGDataset`.
    """

    def __init__(
        self,
        data,
        tmin: float = 0,
        tmax: Optional[float] = None,
        dynamic_items=(),
        output_keys=(),
        **kwargs,
    ):
        dynamic_items = [self._make_load_epoch_dynamic_item(tmin, tmax)] + list(
            dynamic_items
        )
        super().__init__(
            data, dynamic_items=dynamic_items, output_keys=output_keys, **kwargs
        )

    @classmethod
    def json_data_from_bids_path(cls, bids_path) -> dict[str, Any]:
        raw_json_data = super().json_data_from_bids_path(bids_path)

        json_data = {}
        for uid, sample in raw_json_data.items():
            bids_path = get_bids_path_from_fname(sample["fpath"])
            raw = read_raw_bids(
                bids_path, extra_params=dict(preload=False), verbose=False
            )
            stim_channels = mne.utils._get_stim_channel(
                None, raw.info, raise_error=False
            )
            if len(stim_channels) > 0:
                # returns empty array if none found
                events = mne.find_events(raw, shortest_event=0, verbose=False)
                event_id = {}
            else:
                events, event_id = mne.events_from_annotations(
                    raw, verbose=False
                )

            event_id = {v: k for k, v in event_id.items()}

            for onset, _, event in events:
                label = event_id.get(event, int(event))
                event_sample = dict(**sample, label=label, onset=int(onset))
                event_uid = f"{uid}/{label}/{onset}"
                json_data[event_uid] = event_sample

        return json_data

    @classmethod
    def from_moabb(
        cls,
        dataset: BaseMOABBDataset,
        json_path: str | Path,
        subjects=None,
        save_path: str | None = None,
        **cls_kwargs,
    ) -> Self:
        if "tmin" not in cls_kwargs:
            cls_kwargs.update(tmin=0)
        if "tmax" not in cls_kwargs:
            cls_kwargs.update(tmax=dataset.interval[1] - dataset.interval[0])

        return super().from_moabb(
            dataset, json_path, subjects, save_path, **cls_kwargs
        )

    def __getitem__(self, index) -> EpochedEEGSample:
        return super().__getitem__(index)  # type: ignore

    def _make_load_epoch_dynamic_item(self, tmin: float, tmax: Optional[float]):

        @takes("raw", "onset")
        @provides("epoch")
        def _load_epoch(raw: mne.io.RawArray, onset: int):
            # Convert tmin, tmax in seconds to integer indices
            onset_time = onset / raw.info["sfreq"]
            tmin_index = int((onset_time + tmin) * raw.info["sfreq"])
            tmax_index = (
                int((onset_time + tmax) * raw.info["sfreq"])
                if tmax is not None
                else -1
            )

            return raw._getitem(
                (slice(None), slice(tmin_index, tmax_index)),
                return_times=False,
            )

        return _load_epoch


class InMemoryDataset:
    """Wraps a dataset to cache computed items in memory.

    Arguments
    ---------
    dataset : Dataset
        The dataset to delegate to when an item is not available in cache.
    """

    def __new__(cls, dataset: Dataset):
        """Create a new instance of the wrapped dataset."""
        class Wrapper(dataset.__class__):
            """hacking way to perform the cache."""
            def __init__(self):
                self.__wrapped_dataset = dataset
                self.__cache = {}

            def __getitem__(self, index) -> Any:
                if index not in self.__cache:
                    self.__cache[index] = self.__wrapped_dataset[index]
                return self.__cache[index]

            def __dir__(self) -> list[str]:
                """Ensure tab-completion works properly."""
                return dir(self.__wrapped_dataset)

            def __getattr__(self, item):
                return getattr(self.__wrapped_dataset, item)

        return Wrapper()
