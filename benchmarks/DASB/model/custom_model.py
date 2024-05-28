import torch
from torch import nn
from ..utils.hparams import as_list
import math


class AttentionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        att_w = torch.nn.functional.softmax(x, dim=2)
        return att_w


class Discrete_EmbeddingLayer(torch.nn.Module):
    """This class handles embedding layers  for discrete tokens.

    Arguments
    ---------
    num_codebooks: int ,
        number of codebooks of the tokenizer.
    vocab_size : int,
        size of the dictionary of embeddings
    emb_dim: int ,
        the size of each embedding vector
    pad_index: int (default: 0),
        If specified, the entries at padding_idx do not contribute to the gradient.
    init: boolean (default: False):
        If set to True, init the embedding with the tokenizer embedding otherwise init randomly.
    freeze: boolean (default: False)
       If True, the embedding is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.

    Example
    -------
    >>> from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> print(tokens.shape)
    torch.Size([4, 4, 2])
    >>> emb= Discrete_EmbeddingLayer(2, 1024, 1024)
    >>> in_emb = emb(tokens)
    >>> print(in_emb.shape)
    torch.Size([4, 4, 2, 1024])
    """

    def __init__(
        self,
        num_codebooks,
        vocab_size,
        emb_dim,
        pad_index=0,
        init=False,
        freeze=False,
    ):
        super(Discrete_EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.freeze = freeze
        self.embedding = torch.nn.Embedding(
            num_codebooks * vocab_size, emb_dim
        ).requires_grad_(not self.freeze)
        #  TODO: handle padding tokens and initialization with embedding from codec

    def forward(self, in_tokens):
        """Computes the embedding for discrete tokens.
        a sample.

        Arguments
        ---------
        in_tokens : torch.Tensor
            A (Batch x Time x num_codebooks)
            audio sample
        Returns
        -------
        in_embs : torch.Tensor
        """
        with torch.set_grad_enabled(not self.freeze):
            #  Add unique token IDs across diffrent codebooks by adding num_codebooks * vocab_size
            in_tokens += torch.arange(
                0,
                self.num_codebooks * self.vocab_size,
                self.vocab_size,
                device=in_tokens.device,
            )
            # Forward Pass to embedding and
            in_embs = self.embedding(in_tokens)
            return in_embs


class MultiEmbedding(nn.Module):
    """A wrapper module with multiple embedding 'heads' - for
    cases with multiple tokens per sequence

    Arguments
    ---------
    num_embeddings : int
        Size of the dictionary of embeddings.
    embedding_dim : int
        It is the dim of embedding (i.e, the dimensionality of the output).
    num_heads : int
        The number of embedding "heads" (i.e. tokens per step)
    normalized : bool, optional
        Whether to normalize the embeddings (for transformers)
    d_model : int, optional
        The model dimension (igored if not normalized)
    norm_factor : float, optional
        The normalization factor (multiplier)
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        num_heads,
        normalized=False,
        d_model=512,
        norm_factor=None,
    ):
        super().__init__()
        self.emb = torch.nn.ModuleList(
            torch.nn.Embedding(num_embeddings, embedding_dim)
            for _ in range(num_heads)
        )
        self.normalized = normalized
        if norm_factor is None:
            norm_factor = math.sqrt(d_model) if normalized else 1.0
        self.norm_factor = norm_factor

    def forward(self, x):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            A tensor of indexes

        Returns
        -------
        emb : torch.Tensor
            An embedding tensor"""
        emb = (
            torch.cat(
                [
                    emb(x[..., idx].int()).unsqueeze(-2)
                    for idx, emb in enumerate(self.emb)
                ],
                dim=-2,
            )
            * self.norm_factor
        )
        return emb

    def initialize(self, emb):
        """Initializes the embeddings with the specified embedding tensor

        Arguments
        ---------
        emb : torch.Tensor
            A (Layer x Embeddings x Embedding Dim) tensor"""
        with torch.no_grad():
            for head, head_emb in zip(self.emb, emb,):
                head.weight.copy_(head_emb)

    def all_weights(self):
        """Returns all embedding weights as a single tensor"""
        return torch.stack([emb.weight for emb in self.emb])


class HierarchicalUnitWrapper(torch.nn.Module):
    """A wrapper for models similar to UnitHiFiGan that combine multiple layers with offsets

    Arguments
    ---------
    model : torch.nn.Module | Pretrained
        A model
    available_layers : list
        The list of available layers
    num_units : int
        The total number of units/tokens available
    layers : list
        The layers that will be used. If omitted, all layers will be used
    offset : int, optional
        The offset added globally to all layers
    use_length : False
        Whether to use length
    """

    def __init__(
        self, model, available_layers, num_units, layers=None, offset=0, use_length=False
    ):
        super().__init__()
        self.model = model
        self.device = next(iter(param for param in model.parameters())).device
        self.available_layers = as_list(available_layers)
        if layers is None:
            self.layers = self.available_layers
        else:
            self.layers = as_list(layers)
        layers_set = set(self.layers)
        available_layers_set = set(available_layers)
        if not layers_set.issubset(available_layers_set):
            unavailable_layers = ",".join(
                str(layer) for layer in (layers_set - available_layers_set)
            )
            raise ValueError(f"Layers {unavailable_layers} are not supported")
        self.num_units = num_units
        self.layer_offset = self.compute_offset()
        self.offset = offset
        if hasattr(self.model, "tokenize"):
            self.model.tokenize = False
        self.use_length = use_length

    def compute_offset(self):
        """Computes offsets for each layer"""
        _, layers_idx = torch.where(
            torch.tensor(self.available_layers, device=self.device).unsqueeze(0)
            == torch.tensor(self.layers).unsqueeze(1)
        )
        offset = torch.tensor(layers_idx, device=self.device) * self.num_units
        return offset[None, None, :]

    def forward(self, units, length, **kwargs):
        units_with_offset = (
            units + self.layer_offset.to(units.device) + self.offset
        )
        if self.use_length:
            result = self.model(units_with_offset, length, **kwargs)
        else:
            result = self.model(units_with_offset, **kwargs)
        return result


class EncodecVocoder(nn.Module):
    """A vocoder wrapper for Encodec
    
    Arguments
    ---------
    encodec: speechbrain.lobes.models.huggingface_transformers.Encodec
        An Encodec model
    """
    def __init__(self, encodec):        
        super().__init__()
        self.encodec = encodec

    def forward(self, units, length=None):
        """Computes the forward pass"""
        return self.encodec.decode(units, length)