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

def test_ica_method(method: str, n_components: int = 15, **kwargs):
    """Test a specific ICA method and return timing results."""
    print(f"\nTesting ICA method: {method}")
    ica_processor = ICAProcessor(
        n_components=n_components,
        method=method,
        **kwargs
    )
    
    dataset = EpochedEEGDataset.from_moabb(
        BNCI2014_001(),
        f"data/MNE-BIDS-bnci2014-001-epoched-{method}.json",
        save_path="data",
        tmin=0,
        tmax=4.0,
        preload=True,
        output_keys=["label", "subject", "session", "epoch"],
        ica_processor=ica_processor
    )

    # First run - ICA computation
    print("First run (computing ICA):")
    start = time.time()
    for _ in dataset:
        pass
    computation_time = time.time() - start
    print(f"Time with {method} ICA (first run): {computation_time:.2f}s")

    # Second run - using cached ICA
    print("\nSecond run (using cached ICA):")
    start = time.time()
    for _ in dataset:
        pass
    cached_time = time.time() - start
    print(f"Time with {method} ICA (cached): {cached_time:.2f}s")

    # Memory-cached version
    print("\nTesting with InMemoryDataset wrapper:")
    dataset_cached = InMemoryDataset(dataset)
    start = time.time()
    for _ in dataset_cached:
        pass
    memory_cached_time = time.time() - start
    print(f"Time with {method} ICA (in-memory cache): {memory_cached_time:.2f}s")

    return {
        'method': method,
        'computation_time': computation_time,
        'cached_time': cached_time,
        'memory_cached_time': memory_cached_time
    }

def compare_ica_methods():
    # Test without ICA first as baseline
    print("\nTesting without ICA (baseline):")
    dataset_no_ica = EpochedEEGDataset.from_moabb(
        BNCI2014_001(),
        "data/MNE-BIDS-bnci2014-001-epoched.json",
        save_path="data",
        tmin=0,
        tmax=4.0,
        output_keys=["label", "subject", "session", "epoch"],
    )
    
    start = time.time()
    for _ in dataset_no_ica:
        pass
    baseline_time = time.time() - start
    print(f"Time without ICA: {baseline_time:.2f}s")

    # Test different ICA methods
    results = []
    
    # Test Picard
    results.append(test_ica_method(
        'picard',
        n_components=15,
        fit_params={'max_iter': 500}
    ))
    
    # Test Infomax
    results.append(test_ica_method(
        'infomax',
        n_components=15,
        fit_params={'max_iter': 1000}
    ))

    # Print comparison
    print("\nComparison Summary:")
    print("-" * 50)
    print(f"Baseline (no ICA): {baseline_time:.2f}s")
    print("-" * 50)
    for result in results:
        print(f"Method: {result['method']}")
        print(f"  Computation time: {result['computation_time']:.2f}s")
        print(f"  Cached access time: {result['cached_time']:.2f}s")
        print(f"  In-memory cached time: {result['memory_cached_time']:.2f}s")
        print("-" * 50)

@profile
def profile_memory_usage():
    # Profile memory usage for both methods
    for method in ['picard', 'infomax']:
        print(f"\nProfiling {method} ICA:")
        ica_processor = ICAProcessor(
            n_components=15,
            method=method,
            fit_params={'max_iter': 500} if method == 'picard' else {'iteration': 1000}
        )
        dataset = EpochedEEGDataset.from_moabb(
            BNCI2014_001(),
            f"data/MNE-BIDS-bnci2014-001-epoched-{method}.json",
            save_path="data",
            tmin=0,
            tmax=4.0,
            preload=True,
            output_keys=["label", "subject", "session", "epoch"],
            ica_processor=ica_processor
        )

        for _ in dataset:
            pass

if __name__ == "__main__":
    print("Running ICA method comparison...")
    compare_ica_methods()
    
    print("\nRunning memory profile...")
    profile_memory_usage()