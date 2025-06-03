import yaml
import numpy as np
from itertools import product
import random
from hyperpyyaml import load_hyperpyyaml

def load_config_with_search_space(yaml_path):
    """Load YAML config and return (base_config, search_space_dict)"""
    with open(yaml_path) as f:
        config = load_hyperpyyaml(f)  # <- THIS IS THE CORRECT WAY
    base_config = {k: v for k, v in config.items() if k != "search_space"}
    search_space = config.get("search_space", {})
    return base_config, search_space

def load_search_space_only(yaml_path):
    """Extract only the search_space dict from a YAML file (avoids parsing custom tags)."""
    with open(yaml_path) as f:
        lines = f.readlines()
    in_search_space = False
    search_space_lines = []
    for line in lines:
        if line.strip().startswith('search_space:'):
            in_search_space = True
            search_space_lines.append(line)
            continue
        if in_search_space:
            # End if a non-indented line or a new top-level key
            if line.startswith((' ', '\t', '-')) or not line.strip():
                search_space_lines.append(line)
            else:
                break
    # Now parse the collected search_space YAML lines
    search_space_yaml = ''.join(search_space_lines)
    search_space = yaml.safe_load(search_space_yaml)
    if search_space is None:
        return {}
    return search_space.get('search_space', {})

def get_optuna_space(search_space):
    """
    Convert search space dict to Optuna format.
    Returns a dict mapping param names to (suggest_func, *args, **kwargs)
    """
    space = {}
    for key, spec in search_space.items():
        t = spec['type']
        if t == "uniform":
            space[key] = ("suggest_float", spec["min"], spec["max"], spec.get("precision", 4))
        elif t == "discrete_uniform":
            space[key] = ("suggest_int", spec["min"], spec["max"])
        elif t == "choice":
            space[key] = ("suggest_categorical", spec["values"])
        else:
            raise ValueError(f"Unknown type: {t}")
    return space

def get_orion_space(search_space):
    """
    Convert search space dict to Orion CLI format.
    Returns a dict mapping param names to CLI strings.
    """
    cli_space = {}
    for key, spec in search_space.items():
        t = spec['type']
        if t == "uniform":
            cli_space[key] = f'--{key}~"uniform({spec["min"]}, {spec["max"]}, precision={spec.get("precision",4)})"'
        elif t == "discrete_uniform":
            cli_space[key] = f'--{key}~"uniform({spec["min"]}, {spec["max"]}, discrete=True)"'
        elif t == "choice":
            cli_space[key] = f'--{key}~"choices({spec["values"]})"'
        else:
            raise ValueError(f"Unknown type: {t}")
    return cli_space

def generate_grid(search_space):
    """Yield all combos as dicts (for grid search)."""
    keys, values = [], []
    for k, spec in search_space.items():
        t = spec['type']
        if t == "uniform":
            precision = spec.get("precision", 4)
            step = 10 ** -precision
            vals = np.round(np.arange(spec["min"], spec["max"]+step, step), precision).tolist()
        elif t == "discrete_uniform":
            vals = list(range(spec["min"], spec["max"]+1))
        elif t == "choice":
            vals = spec["values"]
        else:
            raise ValueError(f"Unknown type: {t}")
        keys.append(k)
        values.append(vals)
    for combo in product(*values):
        yield dict(zip(keys, combo))

def sample_random(search_space, n_samples):
    """Yield n_samples random combos."""
    for _ in range(n_samples):
        params = {}
        for k, spec in search_space.items():
            t = spec['type']
            if t == "uniform":
                precision = spec.get("precision", 4)
                params[k] = round(random.uniform(spec["min"], spec["max"]), precision)
            elif t == "discrete_uniform":
                params[k] = random.randint(spec["min"], spec["max"])
            elif t == "choice":
                params[k] = random.choice(spec["values"])
        yield params