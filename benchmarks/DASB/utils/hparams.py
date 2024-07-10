"""Utilities for hyperparameter files

Authors
 * Artem Ploujnikov 2024
"""
from numbers import Number

_DTYPE_CONVERT = {"int": int, "float": float}




def repeat_for_layers(layers, value):
    """Repeats the same value for each layer

    Arguments
    ---------
    layers : list | int | str
        a list of layers or a single identifier
    value : object
        the value to repeat

    Returns
    -------
    result : list
        The value repeated for the correct number
        of layers"""
    num_layers = len(layers)
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
