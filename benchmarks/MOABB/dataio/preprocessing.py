"""EEG preprocessing functions.

Authors
 * Drew 2025
 * Bruno 2025
"""
import logging
from functools import cache

import mne
import torch
from speechbrain.utils.data_pipeline import provides, takes


@takes("epoch")
@provides("epoch")
def to_tensor(epoch):
    """Convert a numpy array to a PyTorch tensor."""
    return torch.from_numpy(epoch).float()


# Wrap `create_filter` in a cache so that expensive filters
# will only be created once.
cached_create_filter = cache(mne.filter.create_filter)


@takes("epoch", "info", "target_sfreq", "fmin", "fmax")
@provides("epoch", "sfreq", "target_sfreq", "fmin", "fmax")
def bandpass_resample(epoch, info, target_sfreq, fmin, fmax):
    """Bandpass filter and resample an epoch."""

    bandpass = cached_create_filter(
        None,
        info["sfreq"],
        l_freq=fmin,
        h_freq=fmax,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )

    # Check that filter length is reasonable
    filter_length = len(bandpass)
    len_x = epoch.shape[-1]
    if filter_length > len_x:
        # TODO: These long filters result in massive performance degradation... Do we
        #       want to throw an error instead? This usually happens when fmin is used
        logging.warning(
            "filter_length (%i) is longer than the signal (%i), "
            "distortion is likely. Reduce filter length or filter a longer signal.",
            filter_length,
            len_x,
        )

    yield mne.filter.resample(
        epoch,
        up=target_sfreq,
        down=info["sfreq"],
        method="polyphase",
        window=bandpass,
    )
    yield target_sfreq
