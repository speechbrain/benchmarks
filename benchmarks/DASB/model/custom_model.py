import torch
from torch import nn
from speechbrain.dataio.dataio import length_to_mask
import math
import logging

from huggingface_hub import snapshot_download

try:
    from speechtokenizer import SpeechTokenizer
except ImportError:
    logging.warning("speechtokenizer is not available")


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
        self.init = init

    def init_embedding(self, weights):
        with torch.no_grad():
            self.embedding.weight = torch.nn.Parameter(weights)

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


class DACFeatureExtractor(nn.Module):
    """An adapter for feature extraction

    Arguments
    ---------
    dac : DAC
        a DAC model
    """

    def __init__(self, dac, n_quantizers):
        super().__init__()
        self.dac = dac
        self.dac.eval()
        self.n_quantizers = n_quantizers

    def encode(self, inputs, length):
        """Encodes a raw audio sample using DAC

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            A tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Tokens x Heads) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers

        """
        if inputs.dim() < 3:
            inputs = inputs.unsqueeze(1)
        emb, codes, _, _, _ = self.dac.encode(
            inputs, n_quantizers=self.n_quantizers
        )
        emb.transpose_(1, 2)
        codes.transpose_(1, 2)
        max_len = emb.size(1)
        mask = length_to_mask(
            length * max_len, max_len, device=inputs.device
        ).unsqueeze(-1)
        return codes * mask, emb * mask

    def forward(self, inputs, length):
        """Encodes a raw audio sample using DAC

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            A tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Tokens x Heads) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers

        """
        return self.encode(inputs, length)

    def embeddings(self, tokens):
        """Converts token indexes to vector embeddings

        Arguments
        ---------
        tokens : torch.Tensor
            a (Batch x Length x Heads) tensor of token indexes

        Returns
        -------
        emb : torch.Tensor
            a (Batch x Length x Heads x Embedding) tensor
            of raw vector embeddings from the model's
            quantizer codebooks
        """
        emb, _, _ = self.dac.quantizer.from_codes(tokens.transpose(1, 2).int())
        return emb.transpose(1, 2)


class SpeechTokenizerFeatureExtractor(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained SpeechTokenizer.

    Please, install speechtokenizer:
    pip install speechtokenizer

    Source paper: https://arxiv.org/abs/2308.16692


    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    speech_tokenizer : speechbrain.lobes.models.discrete.speechtokenizer_interface.SpeechTokenizer_interface
        The speech tokenizer interface
    codebooks : int, optional
        The number of codebooks to use - if omitted,
    """

    def __init__(self, speech_tokenizer, codebooks=None):
        super().__init__()
        self.speech_tokenizer = speech_tokenizer
        self.codebooks = codebooks

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        tokens : torch.Tensor
            A tensor of audio tokens
            Shape: (N_q x Batch x Time) by default
            (Batch x Time x N_q) if shape == compat

        """
        return self.encode(wav, wav_lens)

    def encode(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Seq, N_q) tensor of audio tokens

        """
        # Extract discrete codes from SpeechTokenizer
        codes = self.speech_tokenizer.encode(
            wav.unsqueeze(1), wav_lens
        )  # codes: (n_q, B, T)
        if self.codebooks is not None:
            codes = codes[: self.codebooks]
        codes = codes.permute(1, 2, 0)
        return codes

    def decode(self, codes):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        tokens : torch.Tensor
            A (N_q, Batch x Seq) tensor of audio tokens

        Returns
        -------
        wav : torch.Tensor (signal)
            A batch of reconstructed audio signals.
        """
        codes = codes.permute(2, 0, 1)
        return self.speech_tokenizer.decode(codes)
