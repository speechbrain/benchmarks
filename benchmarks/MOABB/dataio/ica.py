from pathlib import Path
from typing import Union

import mne
from mne.preprocessing import ICA


class ICAProcessor:
    """Handles ICA computation and application for EEG data.

    Arguments
    ---------
    n_components : int | float | None
        Number of components to keep during ICA decomposition
    random_state : int | None
        Random state for reproducibility
    """

    def __init__(self, n_components=None, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def get_ica_path(self, raw_path: Union[str, Path]) -> Path:
        """Generate path where ICA solution should be stored."""
        path = Path(raw_path)
        return path.parent / f"{path.stem}_ica.fif"

    def compute_ica(self, raw: mne.io.RawArray, ica_path: Path) -> ICA:
        """Compute ICA solution and save to disk."""
        # High-pass filter for ICA
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=1.0, h_freq=None)

        ica = ICA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        ica.fit(raw)
        ica.save(ica_path)
        return ica


    def process(self, raw: mne.io.RawArray, raw_path: Union[str, Path]) -> mne.io.RawArray:
        """Process raw data with ICA, computing or loading from cache."""
        if not raw.preload:
            raw.load_data()
        
        ica_path = self.get_ica_path(raw_path)
        
        if not ica_path.exists():
            ica = self.compute_ica(raw, ica_path)
        else:
            ica = mne.preprocessing.read_ica(ica_path)
        
        # Create a copy of the raw data before applying ICA
        raw_ica = raw.copy()
        ica.apply(raw_ica)
        
        return raw_ica