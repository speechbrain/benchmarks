"""File for testing ICA computation and application for EEG data.
Authors
-------
Victor Cruz, 2025
"""
import time
import mne
import moabb
import logging
from pathlib import Path
from datetime import datetime
from moabb.datasets import BNCI2014_001
from memory_profiler import profile

from dataio.datasets import EpochedEEGDataset, InMemoryDataset
from dataio.ica import ICAProcessor

# Set up logging
mne.set_log_level(verbose=False)
moabb.set_log_level(level="ERROR")


# Configure logging
def setup_logging():
    """Set up logging to both file and console.

    The logs are written to a file in the 'logs' directory, with a timestamp
    in the filename. The logs are also printed to the console.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"ica_benchmark_{timestamp}.log"

    # Configure logging format
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Set up logger
    logger = logging.getLogger("ICA_benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def test_ica_method(
    method: str, n_components: int = 15, use_hash: bool = True, **kwargs
):
    """Test a specific ICA method and return timing results.

    This function creates an ICAProcessor, runs the EpochedEEGDataset with the
    processor, and measures the time taken for various steps, including initial
    ICA computation, caching, and in-memory caching.

    Arguments
    ---------
    method : str
        The ICA method to test, either 'picard' or 'infomax'.
    n_components : int, optional
        The number of ICA components to use, by default 15.
    use_hash : bool, optional
        Whether to use parameter hashing for caching, by default True.
    **kwargs
        Additional parameters to pass to the ICAProcessor constructor.

    Returns
    -------
    dict
        A dictionary containing the timing results for the tested ICA method.
    """
    logger.info(f"\nTesting ICA method: {method} (use_hash={use_hash})")

    start = time.time()
    ica_processor = ICAProcessor(
        n_components=n_components, method=method, use_hash=use_hash, **kwargs
    )
    time_init = time.time() - start
    logger.info(f"Time to create processor: {time_init:.4f}s")

    start = time.time()
    dataset = EpochedEEGDataset.from_moabb(
        BNCI2014_001(),
        f"data/MNE-BIDS-bnci2014-001-epoched-{method}.json",
        save_path="data",
        tmin=0,
        tmax=4.0,
        preload=True,
        output_keys=["label", "subject", "session", "epoch"],
        dynamic_items=[ica_processor.dynamic_item],
    )
    time_create = time.time() - start
    logger.info(f"Time to create dataset: {time_create:.2f}s")

    # First run - ICA computation
    logger.info("First run (computing ICA):")
    start = time.time()
    for _ in dataset:
        pass
    computation_time = time.time() - start
    logger.info(f"Time with {method} ICA (first run): {computation_time:.2f}s")

    # Second run - using cached ICA
    logger.info("\nSecond run (using cached ICA):")
    start = time.time()
    for _ in dataset:
        pass
    cached_time = time.time() - start
    logger.info(f"Time with {method} ICA (cached): {cached_time:.2f}s")

    # Memory-cached version
    logger.info("\nTesting with InMemoryDataset wrapper:")
    dataset_cached = InMemoryDataset(dataset)
    start = time.time()
    for _ in dataset_cached:
        pass
    memory_cached_time = time.time() - start
    logger.info(
        f"Time with {method} ICA (in-memory cache): {memory_cached_time:.2f}s"
    )

    return {
        "method": method,
        "use_hash": use_hash,
        "init_time": time_init,
        "create_time": time_create,
        "computation_time": computation_time,
        "cached_time": cached_time,
        "memory_cached_time": memory_cached_time,
    }


def compare_ica_methods():
    """Compare the performance of different ICA methods.

    This function tests the Picard and Infomax ICA methods, both with and without
    parameter hashing for caching. It also tests the baseline performance without
    any ICA processing. The results are logged to the console and the log file.
    """
    # Test without ICA first as baseline
    logger.info("\nTesting without ICA (baseline):")
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
    logger.info(f"Time without ICA: {baseline_time:.2f}s")

    # Test different ICA methods
    results = []

    # Test Picard with and without hash
    for use_hash in [True, False]:
        results.append(
            test_ica_method(
                "picard",
                n_components=15,
                use_hash=use_hash,
                fit_params={"max_iter": 500},
                filter_params={"l_freq": 1.0, "h_freq": None},
            )
        )

    # Test Infomax with and without hash
    for use_hash in [True, False]:
        results.append(
            test_ica_method(
                "infomax",
                n_components=15,
                use_hash=use_hash,
                fit_params={"max_iter": 1000},
                filter_params={"l_freq": 1.0, "h_freq": None},
            )
        )

    # Print comparison
    logger.info("\nComparison Summary:")
    logger.info("-" * 70)
    logger.info(f"Baseline (no ICA): {baseline_time:.2f}s")
    logger.info("-" * 70)
    for result in results:
        logger.info(
            f"Method: {result['method']} (use_hash={result['use_hash']})"
        )
        logger.info(f"  Initialization time: {result['init_time']:.4f}s")
        logger.info(f"  Dataset creation time: {result['create_time']:.2f}s")
        logger.info(f"  Computation time: {result['computation_time']:.2f}s")
        logger.info(f"  Cached access time: {result['cached_time']:.2f}s")
        logger.info(
            f"  In-memory cached time: {result['memory_cached_time']:.2f}s"
        )
        logger.info("-" * 70)


@profile
def profile_memory_usage():
    """Profile the memory usage of ICA processing.

    This function runs the ICA processing for both Picard and Infomax methods,
    with and without parameter hashing, and profiles the memory usage.
    """
    # Profile memory usage for both methods with and without hash
    for method in ["picard", "infomax"]:
        for use_hash in [True, False]:
            logger.info(f"\nProfiling {method} ICA (use_hash={use_hash}):")
            ica_processor = ICAProcessor(
                n_components=15,
                method=method,
                use_hash=use_hash,
                fit_params={"max_iter": 500 if method == "picard" else 1000},
                filter_params={"l_freq": 1.0, "h_freq": None},
            )
            dataset = EpochedEEGDataset.from_moabb(
                BNCI2014_001(),
                f"data/MNE-BIDS-bnci2014-001-epoched-{method}.json",
                save_path="data",
                tmin=0,
                tmax=4.0,
                preload=True,
                output_keys=["label", "subject", "session", "epoch"],
                dynamic_items=[ica_processor.dynamic_item],
            )

            for _ in dataset:
                pass


if __name__ == "__main__":
    """Entry point for the ICA benchmark script.

    Runs the ICA method comparison and the memory usage profiling.
    """
    logger.info("Running ICA method comparison...")
    compare_ica_methods()

    logger.info("\nRunning memory profile...")
    profile_memory_usage()
