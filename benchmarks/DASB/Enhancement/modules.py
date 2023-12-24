"""Common modules.

Authors
 * Luca Della Libera 2023
"""

import torch


__all__ = ["MultiHeadEmbedding", "MultiHeadLinear"]


class MultiHeadEmbedding(torch.nn.Module):
    """Multi-head embedding layer.

    Arguments
    ---------
    num_heads:
        The number of heads.
    args:
        The embedding layer positional arguments.
    kwargs:
        The embedding layer keyword arguments.

    Examples
    --------
    >>> import torch
    >>>
    >>>
    >>> model = MultiHeadEmbedding(num_heads=2, num_embeddings=32, embedding_dim=128)
    >>> data = torch.randint(32, size=(8, 6))
    >>> embedding = model(data)

    """

    def __init__(self, num_heads, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.heads = torch.nn.ModuleList(
            [torch.nn.Embedding(*args, **kwargs) for _ in range(num_heads)]
        )

    def forward(self, input):
        """Forward pass.

        Arguments
        ---------
        input:
            The input, shape: ``[B, T]``.

        Returns
        -------
            The output, shape: ``[B, T, embedding_dim]``.

        """
        outputs = [head(input[..., k]) for k, head in enumerate(self.heads)]
        output = torch.stack(outputs).sum(dim=0)
        return output


class MultiHeadLinear(torch.nn.Module):
    """Multi-head linear layer.

    Arguments
    ---------
    num_heads:
        The number of heads.
    args:
        The linear layer positional arguments.
    kwargs:
        The linear layer keyword arguments.

    Examples
    --------
    >>> import torch
    >>>
    >>>
    >>> model = MultiHeadLinear(num_heads=2, in_features=64, out_features=32)
    >>> data = torch.randn(8, 64)
    >>> outputs = model(data)

    """

    def __init__(self, num_heads, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(*args, **kwargs) for _ in range(num_heads)]
        )

    def forward(self, input):
        """Forward pass.

        Arguments
        ---------
        input:
            The input, shape: ``[B, T, C]``.

        Returns
        -------
            The output, shape: ``[B, T, num_heads, C]``.

        """
        outputs = [head(input) for head in self.heads]
        output = torch.stack(outputs).movedim(0, -2)
        return output
