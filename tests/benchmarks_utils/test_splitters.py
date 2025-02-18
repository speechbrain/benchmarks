import pytest
import numpy as np
import logging
from pathlib import Path
from functools import cache

import mne


from dataio.datasets import EpochedEEGDataset

from moabb.datasets import FakeDataset
hparams = dict(target_sampling_frequency=125, fmin=None, fmax=22)

cached_create_filter = cache(mne.filter.create_filter)

@pytest.fixture
def dummy_dataset():
    path = Path("~/mne_data/")

    fake_dataset_folder = path / "MNE-BIDS-Fake"

    if not fake_dataset_folder.exists():
        fake_dataset_folder.mkdir(parents=True)

    dataset = EpochedEEGDataset.from_moabb(
        FakeDataset(n_sessions=2, n_runs=1, n_subjects=3, paradigm="imagery"),
        fake_dataset_folder / "MNE-BIDS-Fake.json",
        save_path=path / "MNE-BIDS-Fake-Epoched",
        tmin=0,
        tmax=4.0,
        output_keys=[
            "label",
            "subject",
            "session",
            "epoch",
        ],
    )
    return dataset


def test_metadata_splitter(dummy_dataset):
    from dataio.splitters import MetadataSplitter
    splitter = MetadataSplitter(dummy_dataset, key="subject")
    # The targets should be the unique subjects
    assert set(splitter.targets) == {"1", "2", "3"}
    # For each target, verify that the test split contains only items with that subject
    for target in splitter.targets:
        split = splitter[target]
        for item in split["test"]:
            assert item["subject"] == target
        # The training split should contain only items not in the test target.
        for item in split["train"]:
            assert item["subject"] != target

def test_leave_k_out_splitter(dummy_dataset):
    from dataio.splitters import LeaveKOutSplitter
    # Using leave_k_out=1, so targets become tuples of one element.
    splitter = LeaveKOutSplitter(dummy_dataset, key="subject", leave_k_out=1)
    expected_targets = {("1",), ("2",), ("3",)}
    assert set(splitter.targets) == expected_targets
    for target in splitter.targets:
        split = splitter[target]
        # Test split: each item's subject must be in the target tuple.
        for item in split["test"]:
            assert item["subject"] in target
        # Train split: should only contain subjects not in the target.
        for item in split["train"]:
            assert item["subject"] not in target


def test_cross_subject_splitter(dummy_dataset):
    from dataio.splitters import CrossSubjectSplitter
    splitter = CrossSubjectSplitter(dummy_dataset, leave_k_out=1)
    expected_targets = {("1",), ("2",), ("3",)}
    assert set(splitter.targets) == expected_targets


def test_cross_session_splitter(dummy_dataset):
    from dataio.splitters import CrossSessionSplitter
    splitter = CrossSessionSplitter(dummy_dataset, leave_k_out=1)
    expected_targets = {("1",), ("2",)} #thanks to last code sprint
    assert set(splitter.targets) == expected_targets

# def test_cross_dataset_splitter(dummy_dataset):
#     from dataio.splitters import CrossDatasetSplitter
#     splitter = CrossDatasetSplitter(dummy_dataset, leave_k_out=1)
#     expected_targets = {("D1",), ("D2",)}
#     assert set(splitter.targets) == expected_targets


def test_metadata_splitter_invalid_dataset():
    from dataio.splitters import MetadataSplitter
    # Create an object that is not an instance of DynamicItemDataset.
    class NotADynamicItemDataset:
        pass

    with pytest.raises(ValueError):
        not_dataset = NotADynamicItemDataset()

        MetadataSplitter(not_dataset, key="subject")



def bandpass_resample(epoch, info):
    bandpass = cached_create_filter(
        None,  # cannot pass epoch (ndarray) if we want to cache the filters
        info["sfreq"],
        l_freq=hparams["fmin"],
        h_freq=hparams["fmax"],
        method="fir",
        fir_design="firwin",
        verbose=False,
    )
    filter_length = len(bandpass)
    len_x = epoch.shape[-1]
    if filter_length > len_x:
        logging.warning(
            "filter_length (%i) is longer than the signal (%i), distortion is likely.",
            filter_length,
            len_x,
        )
    resampled = mne.filter.resample(
        epoch,
        up=hparams["target_sampling_frequency"],
        down=info["sfreq"],
        method="polyphase",
        window=bandpass,
    )
    yield resampled
    yield hparams["target_sampling_frequency"]


def test_bandpass_resample(monkeypatch):
    # Define dummy versions of mne.filter.create_filter and mne.filter.resample.
    dummy_filter = [0] * 10  # Dummy filter of length 10.
    def dummy_create_filter(*args, **kwargs):
        return dummy_filter
    def dummy_resample(epoch, up, down, method, window):
        # Simulate resampling: multiply the epoch by up/down.
        return epoch * (up / down)

    monkeypatch.setattr(mne.filter, "create_filter", dummy_create_filter)
    monkeypatch.setattr(mne.filter, "resample", dummy_resample)
    # Clear and reset the cache for a fresh start.
    global cached_create_filter
    cached_create_filter = cache(mne.filter.create_filter)
    # Create a dummy epoch and info.
    epoch = np.ones((1, 100), dtype=np.float32)
    info = {"sfreq": 50}
    gen = bandpass_resample(epoch, info)
    resampled_epoch = next(gen)
    target_sfreq = next(gen)
    # The expected resampled epoch is epoch * (125/50) = epoch * 2.5.
    np.testing.assert_allclose(resampled_epoch, epoch * 2.5)
    assert target_sfreq == 125
