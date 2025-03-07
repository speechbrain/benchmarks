from pathlib import Path
from typing import Union, Optional, Dict, Any

import mne
from mne.preprocessing import ICA
from mne_bids import get_bids_path_from_fname


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
        method='fastica',
        random_state=42,
        fit_params: Optional[Dict[str, Any]] = None,
        filter_params: Optional[Dict[str, Any]] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.fit_params = fit_params or {}
        self.filter_params = filter_params or {'l_freq': 1.0, 'h_freq': None}

    def get_ica_path(self, raw_path: Union[str, Path]) -> Path:
        """Generate path where ICA solution should be stored.
        
        Creates a derivatives folder to store ICA solutions, following BIDS conventions.
        """
        bids_path = get_bids_path_from_fname(raw_path)
        # For derivatives, you can put them in a derivatives folder:
        bids_path.root = (bids_path.root / ".." / "derivatives" / f"ica-{self.method}")
        # Keep the same base entities:
        bids_path.update(
            suffix='eeg',    # override or confirm suffix
            extension='.fif',
            description='ica',      # <-- This sets a desc=ica entity
            check=True,     # If you do not want BIDSPath to fail on derivative checks
        )
        # Make sure the folder is created
        bids_path.fpath.parent.mkdir(parents=True, exist_ok=True)

        return bids_path.fpath

    def compute_ica(self, raw: mne.io.RawArray, ica_path: Path) -> ICA:
        """Compute ICA solution and save to disk."""
        # High-pass filter for ICA
        raw_filtered = raw.copy()
        raw_filtered.filter(**self.filter_params)

        ica = ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            **self.fit_params
        )
        ica.fit(raw_filtered)
        ica.save(ica_path)
        return ica

    def process(self, raw: mne.io.RawArray, raw_path: Union[str, Path]) -> mne.io.RawArray:
        """Process raw data with ICA, computing or loading from cache."""
        
        ica_path = self.get_ica_path(raw_path)
        
        if not ica_path.exists():
            ica = self.compute_ica(raw, ica_path)
        else:
            ica = mne.preprocessing.read_ica(ica_path, verbose='ERROR')
        
        # Create a copy of the raw data before applying ICA
        raw_ica = raw.copy()
        ica.apply(raw_ica)
        
        return raw_ica
