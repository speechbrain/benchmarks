"""Test module for ICA processing benchmarks.

Authors
-------
Victor Cruz, 2025
"""
import pytest
import time
import mne
import numpy as np
from moabb.datasets import FakeDataset

from dataio.datasets import EpochedEEGDataset
from dataio.ica import ICAProcessor


@pytest.fixture
def dummy_ica_dataset(tmp_path):
    """Create a dummy dataset for testing ICA processing."""
    fake_dataset_folder = tmp_path / "MNE-BIDS-Fake"

    if not fake_dataset_folder.exists():
        fake_dataset_folder.mkdir(parents=True)

    dataset = EpochedEEGDataset.from_moabb(
        FakeDataset(n_sessions=2, n_runs=2, n_subjects=2, paradigm="imagery"),
        fake_dataset_folder / "MNE-BIDS-Fake.json",
        save_path=tmp_path,
        tmin=0,
        tmax=4.0,
        output_keys=["label", "subject", "session", "epoch"],
    )
    return dataset


def test_ica_processor_creation():
    """Test ICA processor initialization."""
    ica_processor = ICAProcessor(
        n_components=15,
        method="picard",
        fit_params={"max_iter": 500},
        filter_params={"l_freq": 1.0, "h_freq": None},
    )
    assert ica_processor.n_components == 15
    assert ica_processor.method == "picard"
    assert ica_processor._fit_params == {"max_iter": 500}


def test_ica_caching(dummy_ica_dataset):
    """Test ICA caching functionality."""
    ica_processor = ICAProcessor(
        n_components=15,
        method="picard",
        fit_params={"max_iter": 500},
        filter_params={"l_freq": 1.0, "h_freq": None},
    )

    # Add ICA processor to dataset
    dataset = dummy_ica_dataset
    dataset.add_dynamic_item(ica_processor.dynamic_item)

    # First run - should compute ICA
    start = time.time()
    for _ in dataset:
        pass
    computation_time = time.time() - start

    # Second run - should use cache
    start = time.time()
    for _ in dataset:
        pass
    cached_time = time.time() - start

    # Cache should be faster
    assert cached_time < computation_time


def test_ica_hash_consistency():
    """Test that ICA hash is consistent for same parameters."""
    ica_processor1 = ICAProcessor(
        n_components=15,
        method="picard",
        fit_params={"max_iter": 500},
        filter_params={"l_freq": 1.0, "h_freq": None},
    )

    ica_processor2 = ICAProcessor(
        n_components=15,
        method="picard",
        fit_params={"max_iter": 500},
        filter_params={"l_freq": 1.0, "h_freq": None},
    )

    # Create dummy raw data
    data = np.random.randn(2, 1000)
    info = mne.create_info(ch_names=["EEG1", "EEG2"], sfreq=100, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    hash1 = ica_processor1._get_params_hash(raw)
    hash2 = ica_processor2._get_params_hash(raw)

    assert hash1 == hash2


def test_different_parameters_different_hash():
    """Test that different ICA parameters produce different hashes."""
    ica_processor1 = ICAProcessor(
        n_components=15,
        method="picard",
        filter_params={"l_freq": 1.0, "h_freq": None},
    )

    ica_processor2 = ICAProcessor(
        n_components=20,  # Different number of components
        method="picard",
        filter_params={"l_freq": 1.0, "h_freq": None},
    )

    # Create dummy raw data
    data = np.random.randn(2, 1000)
    info = mne.create_info(ch_names=["EEG1", "EEG2"], sfreq=100, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    hash1 = ica_processor1._get_params_hash(raw)
    hash2 = ica_processor2._get_params_hash(raw)

    assert hash1 != hash2
