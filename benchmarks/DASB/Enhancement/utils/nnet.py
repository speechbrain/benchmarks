"""Neural network modules.

Authors
 * Luca Della Libera 2023
"""

import torch
from torch import nn


__all__ = ["MultiHeadEmbedding", "MultiHeadLinear"]


class MultiHeadEmbedding(nn.Module):
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
    >>> batch_size = 8
    >>> seq_length = 100
    >>> num_heads = 2
    >>> embedding_dim = 128
    >>> num_classes = 32
    >>> model = MultiHeadEmbedding(num_heads=num_heads, num_embeddings=num_classes, embedding_dim=embedding_dim)
    >>> data = torch.randint(num_classes, size=(batch_size, seq_length, num_heads))
    >>> embedding = model(data)

    """

    def __init__(self, num_heads, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [nn.Embedding(*args, **kwargs) for _ in range(num_heads)]
        )

    def forward(self, input):
        """Forward pass.

        Arguments
        ---------
        input:
            The input, shape: ``[batch_size, seq_length, num_heads]``.

        Returns
        -------
            The output, shape: ``[batch_size, seq_length, embedding_dim]``.

        """
        outputs = [head(input[..., k]) for k, head in enumerate(self.heads)]
        output = torch.stack(outputs).sum(dim=0)
        return output


class MultiHeadLinear(nn.Module):
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
    >>> batch_size = 8
    >>> seq_length = 100
    >>> num_heads = 2
    >>> embedding_dim = 128
    >>> num_classes = 32
    >>> model = MultiHeadLinear(num_heads=num_heads, in_features=embedding_dim, out_features=num_classes)
    >>> data = torch.randn(batch_size, seq_length, embedding_dim)
    >>> outputs = model(data)

    """

    def __init__(self, num_heads, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [nn.Linear(*args, **kwargs) for _ in range(num_heads)]
        )

    def forward(self, input):
        """Forward pass.

        Arguments
        ---------
        input:
            The input, shape: ``[batch_size, seq_length, in_features]``.

        Returns
        -------
            The output, shape: ``[batch_size, seq_length, num_heads, out_features]``.

        """
        outputs = [head(input) for head in self.heads]
        output = torch.stack(outputs).movedim(0, -2)
        return output
