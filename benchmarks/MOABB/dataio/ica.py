"""Module for handling ICA computation and application for EEG data.
Author
------
Victor Cruz, 2025
"""
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
import hashlib

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
        Defaults to {'l_freq': 1.0, 'h_freq': None}
    """

    def __init__(
        self,
        n_components=None,
        method="fastica",
        random_state=42,
        fit_params: Optional[Dict[str, Any]] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        use_hash: bool = True,
    ):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.fit_params = fit_params or {}
        self.filter_params = filter_params or {"l_freq": 1.0, "h_freq": None}
        self.use_hash = use_hash

    def _get_params_hash(self) -> str:
        """Generate a short hash of the ICA parameters."""
        # Select critical parameters that affect the ICA computation
        # not accessible from ICA object for standarization
        critical_params = {
            "n_components": self.n_components,
            "method": self.method,
            "filter_params": self.filter_params,
        }
        # Create a deterministic string representation and hash it
        param_str = json.dumps(critical_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[
            :8
        ]  # First 8 chars are enough

    def get_ica_metadata(self) -> Dict:
        """ Generate metadata dictionary for the ICA parameters. """
        return {
            "n_components": self.n_components,
            "method": self.method,
            "random_state": self.random_state,
            "filter_params": self.filter_params,
            "fit_params": self.fit_params,
        }

    def get_ica_path(self, raw_path: Union[str, Path]) -> tuple[Path, Path]:
        """Generate path where ICA solution should be stored.

        Creates a derivatives folder to store ICA solutions, following BIDS conventions.
        Returns
        -------
        tuple[Path, Path]
            Returns (ica_path, metadata_path)
        """
        bids_path = get_bids_path_from_fname(raw_path)

        if self.use_hash:
            param_hash = self._get_params_hash()
            folder_name = f"ica-{self.method}-{param_hash}"
        else:
            folder_name = f"ica{self.method}"

        # For derivatives, you can put them in a derivatives folder:
        bids_path.root = bids_path.root / ".." / "derivatives" / folder_name
        # Keep the same base entities:
        bids_path.update(
            suffix="eeg",  # override or confirm suffix
            extension=".fif",
            description="ica",  # <-- This sets a desc=ica entity
            check=True,  # If you do not want BIDSPath to fail on derivative checks
        )
        # Make sure the folder is created
        bids_path.fpath.parent.mkdir(parents=True, exist_ok=True)

        ica_path = bids_path.fpath
        metadata_path = ica_path.with_suffix(".json")

        return ica_path, metadata_path

    def save_ica(self, ica: ICA, ica_path: Path, metadata_path: Path):
        """Save ICA solution and metadata to disk."""
        # Save ICA solution
        ica.save(ica_path, overwrite=True)

        # Save metadata
        with metadata_path.open("w") as f:
            json.dump(self.get_ica_metadata(), f)

    def check_ica_metadata(self, metadata_path: Path) -> bool:
        """Check if existing ICA metadata matches current parameters."""
        if not metadata_path.exists():
            return False

        with metadata_path.open() as f:
            saved_metadata = json.load(f)

        current_metadata = self.get_ica_metadata()
        return saved_metadata == current_metadata

    def compute_ica(self, raw: mne.io.RawArray, ica_path: Path) -> ICA:
        """Compute ICA solution and save to disk."""
        # High-pass filter for ICA
        raw_filtered = raw.copy()
        raw_filtered.filter(**self.filter_params)

        ica = ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            **self.fit_params,
        )
        ica.fit(raw_filtered)
        ica.save(ica_path)
        return ica

    @property
    def dynamic_item(self):
        @takes("raw", "fpath")
        @provides("raw", "ica_path")
        def process(raw: mne.io.RawArray, fpath: Union[str, Path]):
            """Process raw data with ICA, computing or loading from cache."""

            ica_path, metadata_path = self.get_ica_path(fpath)

            if ica_path.exists() and self.check_ica_metadata(metadata_path):
                ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
            else:
                ica = self.compute_ica(raw, ica_path)
                self.save_ica(ica, ica_path, metadata_path)

            # Create a copy of the raw data before applying ICA
            raw_ica = raw.copy()
            ica.apply(raw_ica)

            yield raw_ica
            yield ica_path

        return process
