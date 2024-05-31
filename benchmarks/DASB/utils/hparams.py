"""Utilities for hyperparameter files

Authors
 * Artem Ploujnikov 2024
"""

_DTYPE_CONVERT = {"int": int, "float": float}


def as_list(value, dtype=None):
    """Converts comma-separated strings to lists, optionally converting them to lists
    of numbers or other, custom datatypes. Useful for overridable lists of layers
    in hparams

    Arguments
    ---------
    value : object
        the original value
    dtype : str | callable
        "int" for integers
        "float" for floating-point values
        Custom callables are also supported

    Returns
    -------
    value: list
        the provided value, as a list
    """
    if dtype in _DTYPE_CONVERT:
        dtype = _DTYPE_CONVERT[dtype]        
    if dtype and isinstance(value, dtype):
        value = [value]
    else:
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]
        if dtype is not None:
            value = [dtype(item) for item in value]
    return value if isinstance(value, list) else list(value)


def repeat_for_layers(layers, value):
    """Repeats the same value """
    num_layers = layers if isinstance(layers, int) else len(as_list(layers))
    return [value] * num_layers
