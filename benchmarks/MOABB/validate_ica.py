import logging
import os
from pathlib import Path
import time
import mne
import moabb
from moabb.datasets import BNCI2014_001
from memory_profiler import profile

from dataio.datasets import EpochedEEGDataset, RawEEGDataset, InMemoryDataset
from dataio.ica import ICAProcessor  

# Set up logging
mne.set_log_level(verbose=False)
moabb.set_log_level(level="ERROR")

def test_ica_processing():
    # Test without ICA first
    print("\nTesting without ICA:")
    dataset_no_ica = EpochedEEGDataset.from_moabb(
        BNCI2014_001(),
        "data/MNE-BIDS-bnci2014-001-epoched.json",
        save_path="data",
        tmin=0,
        tmax=4.0,
        output_keys=["label", "subject", "session", "epoch"],
    )
    
    # Time iteration
    start = time.time()
    for _ in dataset_no_ica:
        pass
    print(f"Time without ICA: {time.time() - start:.2f}s")

    # Test with ICA
    print("\nTesting with ICA:")
    ica_processor = ICAProcessor(n_components=15)
    dataset_with_ica = EpochedEEGDataset.from_moabb(
        BNCI2014_001(),
        "data/MNE-BIDS-bnci2014-001-epoched-ica.json",
        save_path="data",
        tmin=0,
        tmax=4.0,
        preload=True,
        output_keys=["label", "subject", "session", "epoch"],  # Removed ica_path
        ica_processor=ica_processor
    )

    # First run - ICA computation and caching
    print("First run (computing ICA):")
    start = time.time()
    for _ in dataset_with_ica:
        pass
    print(f"Time with ICA (first run): {time.time() - start:.2f}s")

    # Second run - should use cached ICA
    print("\nSecond run (using cached ICA):")
    start = time.time()
    for _ in dataset_with_ica:
        pass
    print(f"Time with ICA (cached): {time.time() - start:.2f}s")

    # Test with InMemoryDataset wrapper
    print("\nTesting with InMemoryDataset wrapper:")
    dataset_with_ica_cached = InMemoryDataset(dataset_with_ica)
    
    start = time.time()
    for _ in dataset_with_ica_cached:
        pass
    print(f"Time with ICA (in-memory cache): {time.time() - start:.2f}s")

    # Print some sample info
    sample = dataset_with_ica[0]
    print("\nSample info:")
    print(f"Epoch shape: {sample['epoch'].shape}")

@profile
def profile_memory_usage():
    ica_processor = ICAProcessor(n_components=15)
    dataset = EpochedEEGDataset.from_moabb(
        BNCI2014_001(),
        "data/MNE-BIDS-bnci2014-001-epoched-ica.json",
        save_path="data",
        tmin=0,
        tmax=4.0,
        output_keys=["label", "subject", "session", "epoch"],  # Removed ica_path
        ica_processor=ica_processor
    )

    for _ in dataset:
        pass

if __name__ == "__main__":
    print("Running performance tests...")
    test_ica_processing()
    
    print("\nRunning memory profile...")
    profile_memory_usage()