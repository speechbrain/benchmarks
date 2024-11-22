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
