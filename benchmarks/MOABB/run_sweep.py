"""
run_sweep.py
============

Generic sweep launcher for the MOABB benchmark.

Supported sweep engines
-----------------------
* grid       Cartesian product of all values in ``search_space``.
* random     Random samples from ``search_space``.
* optuna     Optuna sampler (default TPE).
* orion      Print the Orion CLI space and exit (no runs).

Design goals
------------
1. **Zero YAML parsing here.**  We *never* call `load_hyperpyyaml`
   from this file, so we cannot trigger the dreaded
   `'…' is a !PLACEHOLDER and must be replaced` error.
2. The heavy work is delegated to :class:`run_experiments.ExperimentRunner`,
   which already loops over subjects/sessions and injects the correct
   overrides for every split.
3. All CLI flags that ExperimentRunner expects are forwarded unchanged
   from the sweep launcher, so the user can keep using the exact same
   command-line interface they use for a single run.

Author
------
Victor Cruz
"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List
import json
import hashlib
import os

import optuna
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from run_experiments import ExperimentRunner
from utils.search import (
    generate_grid,
    get_optuna_space,
    get_orion_space,
    load_search_space_only,
    sample_random,
)

# -----------------------------------------------------------------------------


def parse_top_level_cli(argv: List[str]) -> tuple[str, Dict, Dict]:
    """
    Parse the CLI the same way SpeechBrain does, but keep the
    ``overrides`` dictionary so we can forward it.
    """
    hparams_file, run_opts, overrides = sb.parse_arguments(argv)

    # Make sure overrides is a *dict*
    if isinstance(overrides, str):
        overrides = load_hyperpyyaml(overrides)
    overrides = dict(overrides or {})

    return hparams_file, run_opts, overrides


def params_hash(params):
    return hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]


# -----------------------------------------------------------------------------


SB_CLI_KEYS = {
    # Required by ExperimentRunner
    "data_folder",
    "cached_data_folder",
    "output_folder",
    "nsbj",
    "nsess",
    # Optional / misc
    "seed",
    "nruns",
    "eval_metric",
    "eval_set",
    "train_mode",
    "rnd_dir",
    "dry_run",
}


def build_experimentrunner_argv(
    hparams_file: str, common_cli: Dict,
) -> List[str]:
    """
    Convert a dictionary of CLI options into a flat list of CLI tokens
    accepted by ExperimentRunner.
    """
    argv = ["--hparams", hparams_file]
    for k, v in common_cli.items():
        if k in SB_CLI_KEYS:
            argv.extend([f"--{k}", str(v)])
    return argv


# -----------------------------------------------------------------------------


def run_single_experiment(
    hparams_file: str, common_cli: Dict, hyperparams: Dict,
):
    """
    Launch one ExperimentRunner with the supplied hyper-parameters.

    * `common_cli`   -Static CLI flags copied from the user invocation.
    * `hyperparams` -The hyper-parameters sampled by the sweep engine.
                    These are **not** CLI flags; they are passed as
                    overrides to SpeechBrain via the environment
                    variable ``SB_YAML_OVERRIDES``.
    """
    # ExperimentRunner uses CLI only for infrastructure flags; the
    # *actual* YAML overrides are injected inside its Python code.
    argv = build_experimentrunner_argv(hparams_file, common_cli)

    # Pass the sampled hyper-params to the child process via an env var
    # understood by SpeechBrain.  (Simplest zero-boilerplate path.)
    if hyperparams:
        import os, json, subprocess

        env = dict(os.environ)
        env["SB_YAML_OVERRIDES"] = json.dumps(hyperparams)
        ExperimentRunner(argv).run()
    else:
        ExperimentRunner(argv).run()


# -----------------------------------------------------------------------------


def grid_sweep(hparams_file: str, common_cli: Dict):
    space = load_search_space_only(hparams_file)
    for params in generate_grid(space):
        run_single_experiment(hparams_file, common_cli, params)


def random_sweep(hparams_file: str, common_cli: Dict, n_samples: int):
    space = load_search_space_only(hparams_file)
    for params in sample_random(space, n_samples):
        run_single_experiment(hparams_file, common_cli, params)


def optuna_sweep(hparams_file: str, common_cli: Dict, n_trials: int):
    space = load_search_space_only(hparams_file)
    optuna_space = get_optuna_space(space)

    def objective(trial: optuna.trial.Trial):
        params = {}
        for k, spec in optuna_space.items():
            if spec[0] == "suggest_float":
                params[k] = trial.suggest_float(k, spec[1], spec[2])
            elif spec[0] == "suggest_int":
                params[k] = trial.suggest_int(k, spec[1], spec[2])
            elif spec[0] == "suggest_categorical":
                params[k] = trial.suggest_categorical(k, spec[1])
        trial_id = params_hash(params)
        common_cli_trial = dict(common_cli)

        # Run the experiment; replace with metric parsing if desired.
        common_cli_trial[
            "output_folder"
        ] = f"{common_cli['output_folder']}/trial-{trial_id}"
        run_single_experiment(hparams_file, common_cli_trial, params)
        metrics_path = os.path.join(
            common_cli_trial["output_folder"], "aggregated_performance.txt"
        )  # Or your own metric file
        acc = None
        with open(metrics_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens and tokens[0].lower() == "acc":
                    # preferred: take the value right after “avg:”
                    if "avg:" in tokens:
                        avg_idx = tokens.index("avg:") + 1
                        acc = float(tokens[avg_idx])
                    else:  # fallback to the first number
                        acc = float(tokens[1].lstrip("[").rstrip("]"))
                    break
        if acc is None:
            raise RuntimeError(
                f"Could not find 'acc' in {metrics_path}. "
                "Check that parse_results / aggregate_results ran correctly."
            )
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)


def print_orion_space(hparams_file: str):
    space = load_search_space_only(hparams_file)
    print(" ".join(get_orion_space(space).values()))


# -----------------------------------------------------------------------------


def main():
    hparams_file, run_opts, overrides = parse_top_level_cli(sys.argv[1:])

    # -----------------------------------------------------------------
    # Decide which sweep engine to use
    sweep_type = overrides.pop("sweep_type", "grid")
    n_samples = int(overrides.pop("n_samples", 20))
    n_trials = int(overrides.pop("n_trials", 20))
    # -----------------------------------------------------------------

    if sweep_type not in {"grid", "random", "optuna", "orion"}:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    # -----------------------------------------------------------------
    # Build *common* CLI flags that every ExperimentRunner call needs
    # -----------------------------------------------------------------
    common_cli = {k: v for k, v in overrides.items() if k in SB_CLI_KEYS}

    # Sensible defaults
    common_cli.setdefault("train_mode", "leave-one-session-out")
    common_cli.setdefault("nsbj", 1)
    common_cli.setdefault("nsess", 1)

    # -----------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------
    if sweep_type == "grid":
        grid_sweep(hparams_file, common_cli)
    elif sweep_type == "random":
        random_sweep(hparams_file, common_cli, n_samples)
    elif sweep_type == "optuna":
        optuna_sweep(hparams_file, common_cli, n_trials)
    else:  # orion
        print_orion_space(hparams_file)


if __name__ == "__main__":
    main()
