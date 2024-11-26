"""Braindecode from https://braindecode.org/stable/index.html.
Braindecode is an open-source Python toolbox for decoding raw electrophysiological brain data with 
deep learning models. It includes dataset fetchers, data preprocessing and visualization tools, as 
well as implementations of several deep learning architectures and data augmentations for analysis 
of EEG, ECoG and MEG.

This code is a Speechbrain interface for the Braindecode models. This wrapper allows the usage of 
Braindecode models with the benchmarks pipeline for experiment reproducibility. 

Note 1: The library "einops" is included when braindecode is installed and is not in the package 
requirements of benchmarks or speechbrain. 

Note 2: Softmax is added to the model layer stack since NLL is used.

Authors
 * Davide Borra, 2023
 * Drew Wagner, 2024
 * Victor Cruz, 2024
 * Bruno Aristimunha, 2024
"""
import torch
from einops.layers.torch import Rearrange


class BraindecodeNN(torch.nn.Module):
    """Class for wrapping braindecode models.

    Arguments
    ---------
    model: braindecode.model()
        Braindecode model class

    Example
    -------
    >>> from benchmarks.MOABB.models.EEGConformer import EEGConformer
    >>> model = EEGConformer(input_shape=inp_tensor.shape)
    >>> model_braindecode = BraindecodeNN(model)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_layer = Rearrange("batch time chan 1 -> batch chan time")
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        # (batch, time_, EEG channel, channel) ->  # (batch, EEG channel, time_, channel)
        x = self.input_layer(x)
        x = self.model(x)
        x = self.softmax(x)
        return x
