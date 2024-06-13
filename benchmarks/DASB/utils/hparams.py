"""Utilities for hyperparameter files

Authors
 * Artem Ploujnikov 2024
"""
from numbers import Number

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
    if (
        (dtype is not None and isinstance(value, dtype))
        or isinstance(value, Number)
    ):
        value = [value]
    elif not isinstance(value, list):
        value = list(value)

    return value


def repeat_for_layers(layers, value):
    """Repeats the same value """
    num_layers = layers if isinstance(layers, int) else len(as_list(layers))
    return [value] * num_layers


def choice(value, choices, default=None, apply=False):
    """
    The equivalent of a "switch statement" for hparams files. The typical use case
    is where different options/modules are available, and a top-level flag decides
    which one to use

    Arguments
    ---------
    value: any
        the value to be used as a flag
    choices: dict
        a dictionary maps the possible values of the value parameter
        to the corresponding return values
    default: any
        the default value
    apply: bool
        if set to true, the value is expected to
        be a callable, and the result of the call
        will be returned

    Returns
    -------
    The selected option out of the choices

    Example
    -------
    model: !new:speechbrain.lobes.models.g2p.model.TransformerG2P
        encoder_emb: !apply:speechbrain.utils.hparams.choice
            value: !ref <embedding_type>
            choices:
                regular: !ref <encoder_emb>
                normalized: !ref <encoder_emb_norm>
    """
    if value in choices:
        choice = choices[value]
        if apply and choice is not None:
            choice = choice()
    else:
        choice = default
    return choice
