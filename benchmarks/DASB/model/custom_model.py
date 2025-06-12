import math
import re
import speechbrain as sb
import torch

from speechbrain.nnet.linear import Linear
from model.sq_codec import tokens_to_ternary, ternary_logits_to_tokens
from speechbrain.utils.logger import get_logger


logger = get_logger(__name__)


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
        init=False,
        freeze=False,
        hidden_dim=None,
    ):
        super(Discrete_EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = (
            len(num_codebooks)
            if isinstance(num_codebooks, list)
            else num_codebooks
        )
        self.freeze = freeze
        self.embedding = torch.nn.Embedding(
            self.num_codebooks * vocab_size, emb_dim
        ).requires_grad_(not self.freeze)
        self.init = init

        # Add a linear layer to match dimensions if necessary
        if hidden_dim is not None and hidden_dim != emb_dim:
            self.proj_layer = torch.nn.Linear(emb_dim, hidden_dim)
        else:
            self.proj_layer = None

    def init_embedding(self, weights):
        self.embedding.weight.data.copy_(weights)

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
            if self.proj_layer is not None:
                in_embs = self.proj_layer(in_embs)
            return in_embs


class TernaryPredictionHead(torch.nn.Module):
    """An alternative prediction head that predicts a fixed number of ternary digits
    for each position (as used in SQ-Codec)

    Arguments
    ---------
    d_model : int
        The model dimension
    num_positions : int
        the number of positions
    """
    def __init__(self, d_model, num_positions, d_hidden=512, norm=True):
        super().__init__()
        self.num_positions = num_positions
        self.d_model = d_model
        self.norm = torch.nn.LayerNorm(d_model) if norm else torch.nn.Identity()
        self.lin_hidden = Linear(
            input_size=d_model,
            n_neurons=d_hidden,
        )
        self.act = torch.nn.LeakyReLU()
        self.lin_p = Linear(
            input_size=d_hidden,
            n_neurons=num_positions * 3,
            bias=False
        )

    def forward(self, x, track=None):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            The decoder output (Batch x Length x d_model)

        track : int
            The track index (if applicable)

        Returns
        -------
        p : torch.Tensor
            A tensor of shape (Batch x Length x num_positions x ternary digit)
            The values are logits (unnormalized probabilities)

            p[:, :, :, 0] corresponds to -1
            p[:, :, :, 1] corresponds to 0
            p[:, :, :, 2] corresponds to 1
        """
        batch_size, max_len, _ = x.shape
        x = self.norm(x)
        x = self.lin_hidden(x)
        x = self.act(x)
        p = self.lin_p(x)
        p = p.reshape(batch_size, max_len, self.num_positions, 3)
        return p


class TernaryLogitTokenizer(torch.nn.Module):
    """Converts ternary logits to probabilities

    Arguments
    ---------
    num_positions : int
        The number of ternary digits/positions
    num_tokens : int
        The number of tokens
    chunk_size : int
        The size of the chunk (to prevent OOM)
    mode : str
        "probability" : treats the outputs as a probability distribution
        "argmax" : "hard" mode, only the top probability is used. Cannot be used with
        top_k sampling with k > 1
        
    """
    def __init__(self, num_positions, num_tokens=None, num_tracks=4, chunk_size=10):
        super().__init__()
        self.num_positions = num_positions
        if num_tokens is None:
            num_tokens = 3 ** num_positions
        self.num_tokens = num_tokens
        self.num_tracks = num_tracks
        self.chunk_size = chunk_size
        self.register_buffer("vocab", torch.arange(num_tokens))
        self.register_buffer("vocab_ternary", tokens_to_ternary(self.vocab[None, None, None, :], D=num_positions) + 1)
        self.register_buffer("idx", torch.arange(3)[None, None, None, None, :])

    def forward(self, logits):
        batch_size, max_len, num_positions, _ = logits.shape
        logits = logits.softmax(-1)
        logits = logits.reshape(batch_size, max_len, self.num_tracks, 1, num_positions // self.num_tracks, 3)
        chunks = logits.chunk(
            dim=1,
            chunks=math.ceil(logits.size(1) / self.chunk_size)
        )
        token_logits_chunks = []
        for chunk in chunks:
            token_logits_raw = torch.where(
                self.vocab_ternary[:, None, None, :, :, None] == self.idx,
                chunk,
                torch.ones_like(chunk)
            ).prod(-1).log().sum(-1).exp()
            token_logits_raw_sum = token_logits_raw.sum(-1, keepdim=True)
            token_logits_chunks.append((token_logits_raw / token_logits_raw_sum).squeeze(2))
        token_logits = torch.cat(
            token_logits_chunks,
            dim=1
        )
        return token_logits


@sb.utils.checkpoints.register_checkpoint_hooks
class SaveableGenerator:
    """A wrapper that can be used to store the state of
    the random number generator in a checkpoint. It helps
    with reproducibility in long-running experiments.

    Currently, this only supports CPU and Cuda devices
    natively. If you need training on other architectures,
    consider implementing a custom generator.

    Running it on an unsupported device not using the Torch
    generator interface will simply fail to restore the
    state but will not cause an error.

    Arguments
    ---------
    generators : list, optional
        A list of generator objects. If not provided, 
    """

    def __init__(self, generators=None):
        if generators is None:
            generators = {
                "default": torch.default_generator
            }
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    generators[f"cuda:{idx}"] = _CudaDefaultGeneratorWrapper(idx)

        self.generators = generators

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            key: generator.get_state()
            for key, generator in self.generators.items()
        }
        torch.save(save_dict, path)

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch):
        del end_of_epoch
        save_dict = torch.load(path)
        for key, state in save_dict.items():
            if key == "default":
                torch.default_generator.set_state(state)
                continue
            match = re.match(r"cuda:(\d+)", key)
            if match:
                if not torch.cuda.is_available():
                    logger.warn("Unable to restore RNG for %s, CUDA unavailable", key)
                    continue
                idx = int(match.group(1))
                if idx > torch.cuda.device_count() - 1:
                    logger.warn("Unable to restore RNG for %s, device not found", key)
                    continue
            self.generators[key].set_state(state)


class _CudaDefaultGeneratorWrapper:
    """A generator wrapper for default generators - because torch no longer
    exposes default_generators

    This class should not be used outside of SaveableGenerator

    Arguments
    ---------
    device : int|str
        The device index or identifier"""
    def __init__(self, device):
        self.device = device

    def get_state(self):
        """Returns the generator state

        Returns
        -------
        result : torch.Tensor
            The generator state
        """
        return torch.cuda.get_rng_state(self.device)

    def set_state(self, new_state):
        """"Sets the generator state

        Arguments
        ---------
        new_state : dict
            The new state
        """
        torch.cuda.set_rng_state(new_state, self.device)
