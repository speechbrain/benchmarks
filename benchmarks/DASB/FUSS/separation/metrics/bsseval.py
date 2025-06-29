import numpy as np
import torch
import json
from fast_bss_eval import bss_eval_sources
from speechbrain.utils.metric_stats import MetricStats


__all__ = ["DNSMOS"]


class BSSEval(MetricStats):
    def __init__(self, n_sources, source_names=None, permutation_invariant=True):
        """
        A subclass of MetricStats for evaluating source separation algorithms.

        Args:
            n_sources (int): Number of sources to evaluate.
            source_names (list, optional): Names of the sources. Defaults to None.
            permutation_invariant (bool): Whether to apply permutation invariance when matching sources.
        """
        self.n_sources = n_sources
        self.source_names = source_names or [f"Source {i + 1}" for i in range(n_sources)]
        self.permutation_invariant = permutation_invariant

        # Initialize storage for metrics
        self.metrics = dict()

    def compute_metrics(self, reference_sources, estimated_sources):
        """
        Computes SDR, SIR, and SAR for the given reference and estimated sources.

        Args:
            reference_sources (ndarray): Array of ground truth sources (shape: [n_sources, n_samples]).
            estimated_sources (ndarray): Array of estimated sources (shape: [n_sources, n_samples]).

        Returns:
            dict: A dictionary containing SDR, SIR, and SAR values for each source.
        """
        # Define epsilon
        epsilon = 1e-10
        # Identify rows that are all zeros
        is_all_zeros = torch.all(reference_sources == 0, axis=1)

        # Create a mask to add epsilon only to all-zero rows
        reference_sources[is_all_zeros] += epsilon
        try:
            sdr, sir, sar, perm = bss_eval_sources(reference_sources, estimated_sources, compute_permutation=self.permutation_invariant, load_diag=1e-5)
            is_all_zeros = is_all_zeros[perm]  # Apply permutation to silent mask
            sdr_mean = sdr[~is_all_zeros].mean().detach().cpu().numpy().item()
            sir_mean = sir[~is_all_zeros].mean().detach().cpu().numpy().item()
            sar_mean = sar[~is_all_zeros].mean().detach().cpu().numpy().item()
        except Exception as e: 
            print(f'Exception occured when computing BBSEval: {e}', flush=True)
            sdr_mean, sir_mean, sar_mean = np.nan, np.nan, np.nan
        return {"SDR": sdr_mean, "SIR": sir_mean, "SAR": sar_mean}

    def add(self, reference_sources: torch.Tensor, estimated_sources: torch.Tensor, tag: str = None):
        """
        Adds the metrics for a single evaluation instance.

        Args:
            reference_sources (tensor): Array of ground truth sources (shape: [n_sources, n_samples]).
            estimated_sources (tensor): Array of estimated sources (shape: [n_sources, n_samples]).
        """
        # Ensure inputs are numpy arrays
        reference_sources = reference_sources.squeeze()
        estimated_sources = estimated_sources.squeeze()

        # Validate input shapes
        assert reference_sources.shape[0] == self.n_sources, "Mismatch in number of reference sources."
        assert estimated_sources.shape[0] == self.n_sources, "Mismatch in number of estimated sources."

        # Compute metrics
        metrics = self.compute_metrics(reference_sources, estimated_sources)

        # Store metrics
        for key, values in metrics.items():
            if tag is not None:
                key = f"{key}/{tag}"
            self.metrics.setdefault(key,[]).append(values)

    def summarize(self):
        """
        Summarizes the collected metrics.

        Returns:
            dict: A dictionary containing mean and standard deviation for each metric.
        """
        summary = {}
        for metric, values in self.metrics.items():
            values = np.array(values)
            values = values[~np.isinf(values)]
            summary[metric] = {
                "mean": np.nanmean(values, axis=0).tolist(),
                "std": np.nanstd(values, axis=0).tolist(),
            }

        return summary

    def pretty_print(self):
        """
        Prints the summarized metrics in a human-readable format.
        """
        summary = self.summarize()
        print("Source Separation Evaluation Results:")
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            for i, source_name in enumerate(self.source_names):
                print(f"  {source_name}: Mean = {stats['mean'][i]:.2f}, Std = {stats['std'][i]:.2f}")


    def write_stats(self, path):
        results = self.summarize()
        with open(path, 'w') as outfile:
            json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    n_sources = 2
    source_names = ["Vocals", "Accompaniment"]
    stats = BSSEval(n_sources=n_sources, source_names=source_names, permutation_invariant=True)

    # Example ground truth and estimated sources
    ref_sources = np.random.randn(n_sources, 10000)
    est_sources = np.random.randn(n_sources, 10000)

    stats.add(ref_sources, est_sources)
    stats.pretty_print()
