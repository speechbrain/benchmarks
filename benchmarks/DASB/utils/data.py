"""Data utilities

Authors
 * Artem Ploujnikov 2024
"""

import torch
from speechbrain.dataio.batch import PaddedData
from speechbrain.utils.data_utils import undo_padding


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
        feature = undo_padding(feature.data, feature.lengths)
        feature = [torch.tensor(item, device=device) for item in feature]
    return feature


# TODO: This is not elegant. The original implementation had 
# as_dict() added to PaddedBatch. The benchmark has the limitation
# of not being able to enhance the core.
def as_dict(batch):
    """Converts a batch to a dictionary"""
    return {key: getattr(batch, key) for key in batch._PaddedBatch__keys}
