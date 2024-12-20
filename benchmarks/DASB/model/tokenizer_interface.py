"""
Unified interface for tokenizers, standardizing the output shape of encode and decode functions.

This class reshapes the outputs of various tokenizers to ensure consistency, simplifying integration with recipes and workflows.

Authors
---------
* Pooneh Mousavi, 2024
"""

import torch
from abc import ABC, abstractmethod
from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import (
    DiscreteSSL,
)
from speechbrain.lobes.models.discrete.dac import DAC
from speechbrain.lobes.models.discrete.speechtokenizer_interface import (
    SpeechTokenizer_interface,
)


class BaseTokenizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        """Encode signal into tokens."""
        pass

    @abstractmethod
    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        """Decode tokens to signal."""
        pass

    @abstractmethod
    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size, num_codebooks, device="cpu", **kwargs
    ):
        """Get codebook embeddings."""
        pass


class EncodecTokenizer(Encodec, BaseTokenizer):
    def __init__(self, source, **kwargs):
        Encodec.__init__(self, source=source, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self.encode(signal, lengths)
        if num_codebooks:
            if tokens.shape[-1] < num_codebooks:
                raise ValueError(
                    f"Model only outputs {tokens.shape[-1]} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[..., :num_codebooks]
        return tokens

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        signal = self.decode(tokens)[:, 0]
        return signal

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, device=None, **kwargs
    ):
        embeddings = self.vocabulary
        return embeddings.reshape(-1, embeddings.shape[-1])


class DACTokenizer(DAC, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        DAC.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self(signal[:, None], n_quantizers=num_codebooks)
        return tokens.movedim(-1, -2)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        quantized_feats, _, _ = self.quantizer.from_codes(
            tokens.movedim(-1, -2)
        )
        return self.decode(quantized_feats)[:, 0]

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size, num_codebooks, device="cpu", **kwargs
    ):
        toks = torch.arange(vocab_size, device=device)
        toks = toks[:, None, None].expand(-1, num_codebooks, -1).clone()
        self.to(device).eval()
        z_q, z_p, _ = self.quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)
        z_qs = [
            self.quantizer.quantizers[i].out_proj(z_p_i)
            for i, z_p_i in enumerate(z_ps)
        ]
        return torch.cat(z_qs)[:, :, 0]


class SpeechTokenizer(SpeechTokenizer_interface, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        SpeechTokenizer_interface.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens = self(signal)
        if num_codebooks:
            if len(tokens) < num_codebooks:
                raise ValueError(
                    f"Model only outputs {len(tokens)} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[:num_codebooks]
        return tokens.movedim(-3, -1)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        return self.decode(tokens.movedim(-1, -3))

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size, num_codebooks, device="cpu", **kwargs
    ):
        toks = torch.arange(vocab_size, device=device)
        toks = toks[None, :, None].expand(num_codebooks, -1, -1).clone()
        self.to(device).eval()
        embs = [
            self.model.quantizer.vq.layers[i].decode(indices)
            for i, indices in enumerate(toks)
        ]
        return torch.cat(embs)[:, :, 0]


class DiscreteSSLTokenizer(DiscreteSSL, BaseTokenizer):
    def __init__(self, *args, **kwargs):
        DiscreteSSL.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _, _ = self.encode(signal, lengths)
        if num_codebooks:
            if tokens.shape[-1] < num_codebooks:
                raise ValueError(
                    f"Model only outputs {tokens.shape[-1]} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[..., :num_codebooks]
        return tokens

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        return self.decode(tokens)

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size, num_codebooks, device="cpu", **kwargs
    ):
        toks = torch.arange(vocab_size, device=device)
        toks = toks[None, :, None].expand(num_codebooks, -1, -1).clone()
        self.to(device).eval()
        return torch.cat(
            [self.quantizer.codebooks[i] for i in range(num_codebooks)]
        )
