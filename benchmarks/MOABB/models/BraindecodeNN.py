import torch


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        # (batch, time_, EEG channel, channel) ->  # (batch, EEG channel, time_, channel)
        x = torch.transpose(x, 1, 2)
        x = self.model(x)
        return x
