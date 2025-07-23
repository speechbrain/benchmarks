"""Data utilities

Authors
 * Artem Ploujnikov 2024
"""

import torch
from speechbrain.dataio.batch import PaddedData


def undo_batch(batch):
    """Converts a padded batch or a dicitionary to a list of
    dictionaries. Any instances of PaddedData encountered will
    be converted to plain tensors

    Arguments
    ---------
    batch: dict|speechbrain.dataio.batch.PaddedBatch
        the batch

    Returns
    -------
    result: dict
        a list of dictionaries with each dictionary as a batch
        element
    """
    if hasattr(batch, "as_dict"):
        batch = batch.as_dict()
    keys = batch.keys()
    return [
        dict(zip(keys, item))
        for item in zip(
            *[_unpack_feature(feature) for feature in batch.values()]
        )
    ]


def _unpack_feature(feature):
    """Un-batches a single feature. If a PaddedBatch is provided, it will be converted
    to a list of unpadded tensors. Otherwise, it will be returned unmodified

    Arguments
    ---------
    feature : any
        The feature to un-batch
    """
    if isinstance(feature, PaddedData):
        device = feature.data.device
        feature = _undo_padding(feature.data, feature.lengths)
        feature = [torch.tensor(item, device=device) for item in feature]
    return feature


# NOTE: Similar to the function in speechbrain.utils.data_utils
# but it keeps values in tensor form
def _undo_padding(batch, lengths):
    """Produces Python lists given a batch of sentences with
    their corresponding relative lengths.

    Arguments
    ---------
    batch : torch.Tensor
        Batch of sentences gathered in a batch.
    lengths : torch.Tensor
        Relative length of each sentence in the batch.

    Returns
    -------
    as_list : list
        A python list of the corresponding input tensor.

    Example
    -------
    >>> batch=torch.rand([4,100])
    >>> lengths=torch.tensor([0.5,0.6,0.7,1.0])
    >>> snt_list=undo_padding(batch, lengths)
    >>> len(snt_list)
    4
    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq[:actual_size]
        as_list.append(seq_true)
    return as_list


def as_dict(batch):
    """Converts a batch to a dictionary"""
    return {key: getattr(batch, key) for key in batch._PaddedBatch__keys}
