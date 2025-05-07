"""
Script to run leave-one-subject-out and/or leave-one-session-out training, optionally with multiple seeds.
This script loops over the different subjects and sessions and trains different models.
At the end, the final performance is computed with the aggregate_results.py script that provides the average performance.

Usage:
python run_experiments.py --hparams=hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data \
--output_folder=results/MotorImagery/BNCI2014001/EEGNet --nsbj=9 --nsess=2 --seed=1986 --nruns=2 --number_of_epochs=10


Authors
-------
Victor Cruz, 2025
"""

import sys
import subprocess
from pathlib import Path
import argparse
import random

# import logging
# from typing import Optional
import string

# from utils.exceptions import DryRunComplete


class ExperimentRunner:
    """Manages multiple MOABB experiment runs."""

    def __init__(self, args: list):
        self.args = self.validate_args(args)
        self.setup_experiment()

    def validate_args(self, args) -> argparse.Namespace:
        """Validate and parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Run multiple MOABB experiments"
        )

        # Required arguments
        parser.add_argument(
            "--hparams", required=True, help="Path to hyperparameter file"
        )
        parser.add_argument(
            "--data_folder", required=True, help="Path to data directory"
        )
        parser.add_argument(
            "--output_folder", required=True, help="Path to output directory"
        )
        parser.add_argument(
            "--nsbj", type=int, required=True, help="Number of subjects"
        )
        parser.add_argument(
            "--nsess", type=int, required=True, help="Number of sessions"
        )

        # Optional arguments
        parser.add_argument("--cached_data_folder", help="Path to cached data")
        parser.add_argument("--seed", type=int, help="Random seed")
        parser.add_argument(
            "--nruns", type=int, default=1, help="Number of runs"
        )
        parser.add_argument(
            "--eval_metric", default="acc", help="Evaluation metric (acc, f1)"
        )
        parser.add_argument(
            "--eval_set",
            default="test",
            choices=["test", "dev"],
            help="Evaluation set",
        )
        parser.add_argument(
            "--train_mode",
            default="leave-one-session-out",
            choices=["leave-one-session-out", "leave-one-subject-out"],
            help="Training mode",
        )
        parser.add_argument(
            "--rnd_dir",
            type=bool,
            default=False,
            help="Use random directory name",
        )
        parser.add_argument(
            "--dry_run",
            type=bool,
            default=False,
            help="Validate setup without running",
        )

        args = parser.parse_args(args)

        # Validate arguments
        if args.eval_set == "dev":
            args.metric_file = "valid_metrics.pkl"
        else:
            args.metric_file = "test_metrics.pkl"

        if args.seed is None:
            args.seed = random.randint(0, 99999)

        if not args.cached_data_folder:
            args.cached_data_folder = str(Path(args.data_folder) / "cache")

        return args

    def setup_experiment(self):
        """Setup experiment directories and logging."""
        # Setup random directory if requested
        if self.args.rnd_dir:
            rnd_name = "".join(random.choices(string.ascii_letters, k=6))
            self.args.output_folder = str(
                Path(self.args.output_folder) / rnd_name
            )

        # Create directories
        Path(self.args.output_folder).mkdir(parents=True, exist_ok=True)
        Path(self.args.data_folder).mkdir(parents=True, exist_ok=True)
        Path(self.args.cached_data_folder).mkdir(parents=True, exist_ok=True)

        # Save configuration
        self.save_configuration()

    def save_configuration(self):
        """Save experiment configuration."""
        config_file = Path(self.args.output_folder) / "flags.txt"
        with open(config_file, "w") as f:
            for key, value in vars(self.args).items():
                f.write(f"{key}: {value}\n")

    def run_experiment(self, target_session_idx: int, output_folder_exp: Path):
        """Run experiments for all subjects."""
        # try:
        for target_subject_idx in range(self.args.nsbj):
            print(f"Subject {target_subject_idx}")

            cmd = [
                "python",
                "train.py",
                self.args.hparams,
                f"--seed={self.args.seed}",
                f"--data_folder={self.args.data_folder}",
                f"--cached_data_folder={self.args.cached_data_folder}",
                f"--output_folder={output_folder_exp}",
                f"--target_subject_idx={target_subject_idx}",
                f"--target_session_idx={target_session_idx}",
                f"--data_iterator_name={self.args.train_mode}",
            ]
            print(cmd)

            subprocess.run(cmd)


    def parse_results(self, output_folder_exp: Path, run_name: str):
        """Parse results for current run."""
        cmd = [
            "python",
            "utils/parse_results.py",
            str(output_folder_exp),
            self.args.metric_file,
            self.args.eval_metric,
        ]

        with open(
            Path(self.args.output_folder) / f"{run_name}_results.txt", "a"
        ) as f:
            subprocess.run(cmd, stdout=f)

    def aggregate_final_results(self):
        """Aggregate results across all runs."""
        cmd = [
            "python",
            "utils/aggregate_results.py",
            self.args.output_folder,
            self.args.eval_metric,
        ]

        with open(
            Path(self.args.output_folder) / "aggregated_performance.txt", "a"
        ) as f:
            subprocess.run(cmd, stdout=f)

    def run(self):
        """Execute all experiment runs."""
        # try:
        for run_idx in range(self.args.nruns):
            run_name = f"run{run_idx + 1}"
            output_folder_exp = (
                Path(self.args.output_folder) / run_name / str(self.args.seed)
            )

            if self.args.train_mode == "leave-one-subject-out":
                self.run_experiment(0, output_folder_exp)
            elif self.args.train_mode == "leave-one-session-out":
                for sess_idx in range(self.args.nsess):
                    self.run_experiment(sess_idx, output_folder_exp)

            # Store results
            self.parse_results(output_folder_exp, run_name)

            # Update seed for next run
            self.args.seed += 1

        # Final aggregation
        self.aggregate_final_results()
        # except DryRunComplete:
        #    print("Dry run validation completed successfully")
        #    return True
        # except Exception as e:
        #    if self.args.dry_run:
        #        print(f"Dry run failed: {str(e)}")
        #        return False
        #    raise


if __name__ == "__main__":
    runner = ExperimentRunner(sys.argv[1:])
    sys.exit(0 if runner.run() else 1)
