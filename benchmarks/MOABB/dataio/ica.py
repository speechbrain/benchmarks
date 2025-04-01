"""Module for handling ICA computation and application for EEG data.
Author
------
Victor Cruz, 2025
"""
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
import hashlib
from datetime import datetime

import mne
from mne.preprocessing import ICA
from mne_bids import get_bids_path_from_fname

from speechbrain.utils.data_pipeline import provides, takes


class ICAProcessor:
    """Handles ICA computation and application for EEG data.

    Arguments
    ---------
    n_components : int | float | None
        Number of components to keep during ICA decomposition
    method : str
        The ICA method to use. Can be 'fastica', 'infomax' or 'picard'.
        Defaults to 'fastica'.
    random_state : int | None
        Random state for reproducibility
    fit_params : dict | None
        Additional parameters to pass to the ICA fit method.
        See mne.preprocessing.ICA for details.
    filter_params : dict | None
        Parameters for the high-pass filter applied before ICA.
        Set to None to skip filtering if data is already filtered.
        Defaults to {'l_freq': 1.0, 'h_freq': None}

    Example
    -------
    >>> raw = mne.io.RawArray(data, info)  # Create some MNE raw data
    >>> ica_processor = ICAProcessor(
    ...     n_components=15,
    ...     method="picard",
    ...     fit_params={"max_iter": 500}
    ... )
    >>> # Use in a SpeechBrain pipeline
    >>> # Dynammic item to be used in pipeline: ica_processor.dynamic_item
    """

    def __init__(
        self,
        n_components=None,
        method="fastica",
        random_state=42,
        fit_params: Optional[Dict[str, Any]] = None,
        filter_params: Optional[Dict[str, Any]] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self._fit_params = fit_params or {}
        self.filter_params = filter_params or {"l_freq": 1.0, "h_freq": None}

    def _get_effective_filter_params(self, raw: mne.io.RawArray) -> Dict:
        """Determine effective filtering parameters considering both data and processing.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data.

        Returns
        -------
        dict
            Effective filter parameters considering both intrinsic and applied filters.
        """
        # Get the intrinsic highpass from the data
        data_highpass = raw.info["highpass"]

        # Determine effective highpass
        if self.filter_params and "l_freq" in self.filter_params:
            # If we're applying additional filtering, effective highpass is the higher value
            effective_highpass = max(
                data_highpass, self.filter_params["l_freq"]
            )
        else:
            effective_highpass = data_highpass

        # Similarly for lowpass
        data_lowpass = raw.info["lowpass"]
        if self.filter_params and "h_freq" in self.filter_params:
            # For lowpass, take the lower value if we're applying additional filtering
            effective_lowpass = (
                min(data_lowpass, self.filter_params["h_freq"])
                if self.filter_params["h_freq"]
                else data_lowpass
            )
        else:
            effective_lowpass = data_lowpass

        return {
            "effective_highpass": effective_highpass,
            "effective_lowpass": effective_lowpass,
            "original_data_highpass": data_highpass,
            "original_data_lowpass": data_lowpass,
            "additional_filtering": bool(self.filter_params),
            "filter_params": self.filter_params,
        }

    def _get_data_params(self, raw: mne.io.RawArray) -> Dict:
        """Extract relevant parameters from raw.info and processing.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data.

        Returns
        -------
        dict
            Dictionary containing relevant data parameters.
        """
        filter_info = self._get_effective_filter_params(raw)

        return {
            "effective_highpass": filter_info["effective_highpass"],
            "effective_lowpass": filter_info["effective_lowpass"],
            "sfreq": raw.info["sfreq"],
            "n_channels": len(raw.info["ch_names"]),
            "filtering_applied": filter_info["additional_filtering"],
        }

    def _get_ica_params(self) -> Dict:
        """Get ICA-specific processing parameters.

        Returns
        -------
        dict
            Dictionary containing ICA processing parameters.
        """
        return {
            "n_components": self.n_components,
            "method": self.method,
            "random_state": self.random_state,
            "fit_params": self._fit_params,
            "filter_params": self.filter_params,
        }

    def _get_params_hash(self, raw: mne.io.RawArray) -> str:
        """Generate hash based on effective parameters.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data.

        Returns
        -------
        str
            8-character hexadecimal hash of the parameters.
        """
        filter_info = self._get_effective_filter_params(raw)

        hash_params = {
            "data_params": {
                "effective_highpass": filter_info["effective_highpass"],
                "effective_lowpass": filter_info["effective_lowpass"],
                "sfreq": raw.info["sfreq"],
                "n_channels": len(raw.info["ch_names"]),
            },
            "ica_params": {
                "n_components": self.n_components,
                "method": self.method,
                "random_state": self.random_state,
                "fit_params": self._fit_params,
            },
            "filter_params": filter_info["filter_params"],
        }
        param_str = json.dumps(hash_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def get_ica_metadata(self, raw: mne.io.RawArray) -> Dict:
        """Generate complete metadata including effective parameters.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data.

        Returns
        -------
        dict
            Complete metadata dictionary.
        """
        filter_info = self._get_effective_filter_params(raw)

        return {
            "data_params": self._get_data_params(raw),
            "ica_params": self._get_ica_params(),
            "filter_info": filter_info,
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "raw_filename": str(raw.filenames[0])
                if raw.filenames
                else None,
            },
        }

    def get_ica_path(
        self, raw: mne.io.RawArray, raw_path: Union[str, Path]
    ) -> tuple[Path, Path]:
        """Generate path where ICA solution should be stored.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data.
        raw_path : str | Path
            Path to the raw data file.

        Returns
        -------
        tuple[Path, Path]
            - Path to ICA solution file
            - Path to metadata JSON file
        """
        bids_path = get_bids_path_from_fname(raw_path)

        param_hash = self._get_params_hash(raw)
        folder_name = f"ica-{self.method}-{param_hash}"
        desc = f"ica{param_hash}"

        # For processors, you can put them in a processors folder:
        bids_path.root = bids_path.root / ".." / "processors" / folder_name

        # Keep the same base entities:
        bids_path.update(
            suffix="eeg", extension=".fif", description=desc, check=True,
        )

        # Make sure the folder is created
        bids_path.fpath.parent.mkdir(parents=True, exist_ok=True)

        ica_path = bids_path.fpath
        metadata_path = ica_path.with_suffix(".json")

        return ica_path, metadata_path

    def save_ica(
        self,
        ica: ICA,
        ica_path: Path,
        metadata_path: Path,
        raw: mne.io.RawArray,
    ):
        """Save ICA solution and metadata to disk.

        Arguments
        ---------
        ica : mne.preprocessing.ICA
            The ICA solution to save.
        ica_path : Path
            Path where to save the ICA solution.
        metadata_path : Path
            Path where to save the metadata JSON.
        raw : mne.io.RawArray
            The raw EEG data used for ICA.

        Returns
        -------
        None
        """
        # Save ICA solution
        ica.save(ica_path, overwrite=True)

        # Save metadata including data parameters
        metadata = self.get_ica_metadata(raw)
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

    def check_ica_metadata(
        self, raw: mne.io.RawArray, metadata_path: Path
    ) -> bool:
        """Check if existing ICA metadata matches current parameters.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data to check against.
        metadata_path : Path
            Path to the metadata JSON file.

        Returns
        -------
        bool
            True if metadata exists and matches both data and ICA parameters.
        """
        if not metadata_path.exists():
            return False

        with metadata_path.open() as f:
            saved_metadata = json.load(f)

        # Check data parameters
        current_data_params = self._get_data_params(raw)
        if saved_metadata["data_params"] != current_data_params:
            return False

        # Check ICA parameters
        current_ica_params = self._get_ica_params()
        if saved_metadata["ica_params"] != current_ica_params:
            return False

        return True

    def compute_ica(self, raw: mne.io.RawArray, ica_path: Path) -> ICA:
        """Compute ICA solution considering effective filtering.

        Arguments
        ---------
        raw : mne.io.RawArray
            The raw EEG data to process.
        ica_path : Path
            Path where to save the computed ICA solution.

        Returns
        -------
        mne.preprocessing.ICA
            The computed ICA solution.
        """
        filter_info = self._get_effective_filter_params(raw)

        # Only apply additional filtering if needed
        if filter_info["additional_filtering"]:
            raw_filtered = raw.copy()
            raw_filtered.filter(**self.filter_params)
        else:
            raw_filtered = raw

        ica = ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            **self._fit_params,
        )
        ica.fit(raw_filtered)
        return ica

    @property
    def dynamic_item(self):
        """Creates a dynamic pipeline item for ICA processing.

        Arguments
        ---------
        None
            Uses instance methods and attributes.

        Returns
        -------
        callable
            A function that:
            Takes:
                - raw (mne.io.RawArray): The raw EEG data
                - fpath (Union[str, Path]): Path to the raw data file
            Provides:
                - raw (mne.io.RawArray): The ICA-processed EEG data
                - ica_path (Path): Path to the saved ICA solution
        """

        @takes("raw", "fpath")
        @provides("raw", "ica_path")
        def process(raw: mne.io.RawArray, fpath: Union[str, Path]):
            """Process raw data with ICA, computing or loading from cache."""

            ica_path, metadata_path = self.get_ica_path(raw, fpath)

            if ica_path.exists() and self.check_ica_metadata(
                raw, metadata_path
            ):
                ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
            else:
                ica = self.compute_ica(raw, ica_path)
                self.save_ica(ica, ica_path, metadata_path, raw)

            # Create a copy of the raw data before applying ICA
            raw_ica = raw.copy()
            ica.apply(raw_ica)

            yield raw_ica
            yield ica_path

        return process
