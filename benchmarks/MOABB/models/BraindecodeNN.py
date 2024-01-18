import torch


class BraindecodeNN(torch.nn.Module):
    """Class for wrapping braindecode models.

    Arguments
    ---------
    model: braindecode.model()
        Braindecode model class

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGConformer(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
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
