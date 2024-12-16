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
    @abstractmethod
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, **kwargs):
        """Abstract method to encode a signal into tokens."""
        pass

    @abstractmethod
    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        """Abstract method to decode tokens into a signal."""
        pass

    @abstractmethod
    @torch.no_grad()
    def get_pretrained_embeddings(self, **kwargs):
        """Return pretrained codebook embedding."""
        pass


class EncodecTokenizer(Encodec, BaseTokenizer):
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, **kwargs):
        # signal: [B, T]
        self.eval()
        tokens, _ = self.encode(signal, lengths)  # [B, T, N_Q]
        return tokens

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        # tokens: [B, T, N_Q]
        self.eval()
        signal = self.decode(tokens)[:, 0]  # [B, T]
        return signal

    @torch.no_grad()
    def get_pretrained_embeddings(self, **kwargs):
        """Return pretrained codebook embedding."""
        embeddings = self.vocabulary
        return embeddings.reshape(-1, embeddings.shape[-1])


class DACTokenizer(DAC, BaseTokenizer):
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, **kwargs):
        # signal: [B, T]
        self.eval()
        tokens, _ = self(
            signal[:, None], n_quantizers=kwargs["num_codebooks"]
        )  # [B, N_Q, T]
        return tokens.movedim(-1, -2)  # [B, T, N_Q]

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        # tokens: [B, T, N_Q]
        self.eval()
        quantized_feats, _, _ = self.quantizer.from_codes(
            tokens.movedim(-1, -2)  # [B, N_Q, T]
        )
        signal = self.decode(quantized_feats)[:, 0]  # [B, T]
        return signal

    @torch.no_grad()
    def get_pretrained_embeddings(self, **kwargs):
        """Return pretrained codebook embedding."""
        # See https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py#L200
        toks = torch.arange(kwargs["vocab_size"], device=kwargs["device"])
        toks = (
            toks[:, None, None].expand(-1, kwargs["num_codebooks"], -1).clone()
        )  # [C, K, 1]
        self.to(kwargs["device"]).eval()
        with torch.no_grad():
            z_q, z_p, _ = self.quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)
        z_qs = []
        for i, z_p_i in enumerate(z_ps):
            with torch.no_grad():
                z_q_i = self.quantizer.quantizers[i].out_proj(
                    z_p_i
                )  # [C, H, 1]
            z_qs.append(z_q_i)
        assert (z_q == sum(z_qs)).all()
        embeddings = torch.cat(z_qs)[:, :, 0]
        return embeddings


class SpeechTokenizer(SpeechTokenizer_interface, BaseTokenizer):
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths, **kwargs):
        # signal: [B, T]
        self.eval()
        tokens = self(signal)[: kwargs["num_codebooks"]]  # [N_Q, B, T]
        return tokens.movedim(-3, -1)  # [B, T, N_Q]

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        # tokens: [B, T, N_Q]
        self.eval()
        tokens = tokens.movedim(-1, -3)  # [N_Q, B, T]
        return self.decode(tokens)  # [B, T]

    @torch.no_grad()
    def get_pretrained_embeddings(self, **kwargs):
        """Return pretrained codebook embedding."""
        # See https://github.com/ZhangXInFD/SpeechTokenizer/blob/a9f88dc72642b600654a62861e34342babae6c71/speechtokenizer/quantization/core_vq.py#L360
        toks = torch.arange(kwargs["vocab_size"], device=kwargs["device"])
        toks = (
            toks[None, :, None].expand(kwargs["num_codebooks"], -1, -1).clone()
        )  # [K, C, 1]
        self.to(kwargs["device"]).eval()
        embs = []
        for i, indices in enumerate(toks):
            layer = self.model.quantizer.vq.layers[i]
            with torch.no_grad():
                quantized = layer.decode(indices)
            embs.append(quantized)
        assert (self.model.quantizer.decode(toks) == sum(embs)).all()
        embeddings = torch.cat(embs)[:, :, 0]
        return embeddings


class DiscreteSSLTokenizer(DiscreteSSL, BaseTokenizer):
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths):
        pass

    @torch.no_grad()
    def tokens_to_sig(self, tokens):
        pass

    @torch.no_grad()
    def get_pretrained_embeddings(self, **kwargs):
        pass
