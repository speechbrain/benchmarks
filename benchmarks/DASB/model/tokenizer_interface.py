
"""
Unified interface for tokenizers, standardizing the output shape of encode and decode functions.

This class reshapes the outputs of various tokenizers to ensure consistency, simplifying integration with recipes and workflows.

Authors
---------
* Pooneh Mousavi, 2024
"""

import torch

from  speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
from  speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
from  speechbrain.lobes.models.discrete.dac import DAC
from  speechbrain.lobes.models.discrete.speechtokenizer_interface import SpeechTokenizer_interface


class Tokenizer_Encodec(Encodec):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens,**kwargs):
        # sig: [B, T]
        self.eval()
        toks, _ = self.encode(sig, lens)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks,**kwargs):
        # toks: [B, N, K]
        self.eval()
        sig = self.decode(toks)[:, 0]  # [B, T]
        return sig
  
class Tokenizer_DAC(DAC):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens,**kwargs):
        # sig: [B, T]
        self.eval()
        toks, _ = self(
            sig[:, None], n_quantizers=kwargs['num_codebooks']
        )  # [B, K, N]
        toks = toks.movedim(-1, -2)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks,**kwargs):
        # toks: [B, N, K]
        self.eval()
        qfeats, _, _ = self.quantizer.from_codes(
            toks.movedim(-1, -2)  # [B, K, N]
        )
        sig = self.decode(qfeats)[:, 0]  # [B, T]
        return sig

class Tokenizer_SpeechTokenizer(SpeechTokenizer_interface):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens,**kwargs):
        # sig: [B, T]
        self.eval()
        toks = self(sig)[
            : kwargs['num_codebooks']
        ]  # [K, B, N]
        toks = toks.movedim(-3, -1)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks,**kwargs):
        # toks: [B, N, K]
        self.eval()
        toks = toks.movedim(-1, -3)  # [K, B, N]
        sig = self.decode(toks)  # [B, T]
        return sig

class Tokenizer_DiscreteSSL(DiscreteSSL):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        # sig: [B, T]
        self.hparams.codec_quantizer.to(self.device).eval()
        toks, _, _ = self.hparams.codec_quantizer(
            sig,
            lens,
            SSL_layers=self.hparams.SSL_layers,
            deduplicates=[False] * len(self.hparams.SSL_layers),
            bpe_tokenizers=[None] * len(self.hparams.SSL_layers),
        )  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        # toks: [B, N, K]
        self.hparams.codec_vocoder.device = self.device
        self.hparams.codec_vocoder.to(self.device).eval()

        # Add offset for embedding layer
        all_layer_ids = self.hparams.codec_quantizer.ssl_layer_ids
        # TODO: remove after testing
        assert tuple(all_layer_ids) == (1, 3, 7, 12, 18, 23)
        offsets = torch.arange(
            0,
            len(all_layer_ids) * self.hparams.vocab_size,
            self.hparams.vocab_size,
            device=self.device,
        )
        offset_idxes = [all_layer_ids.index(x) for x in self.hparams.SSL_layers]
        offsets = offsets[offset_idxes]
        toks = toks + offsets + 1

        # Handle missing codebooks
        if len(self.hparams.SSL_layers) < len(all_layer_ids):
            full_toks = torch.zeros(
                *toks.shape[:2],
                len(all_layer_ids),
                dtype=toks.dtype,
                device=self.device,
            )
            for i, idx in enumerate(offset_idxes):
                full_toks[..., idx] = toks[..., i]
            toks = full_toks

        self.hparams.codec_vocoder.tokenize = False
        sig = self.hparams.codec_vocoder(toks)[:, 0]  # [B, T]
        return sig

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self,sig, lens,**kwargs):
        toks = self.tokenizer.sig_to_toks(sig, lens,**kwargs)
        return toks
    
    @torch.no_grad()
    def decode(self,sig,**kwargs):
        sig = self.tokenizer.toks_to_sig(sig,**kwargs)
        return sig
    
    
# model_hub = "facebook/encodec_24khz"
# save_path = "savedir"
# model = Tokenizer_Encodec(model_hub, save_path)
# from speechbrain.lobes.models.huggingface_transformers.hubert import (HuBERT)
# inputs = torch.rand([3, 2000])
# model_hub = "facebook/hubert-large-ll60k"
# save_path = "savedir"
# ssl_layer_num = [7,23]
# deduplicate =[False, True]
# bpe_tokenizers=[None, None]
# kmeans_repo_id = "speechbrain/SSL_Quantization"
# kmeans_dataset = "LJSpeech"
# num_clusters = 1000
# ssl_model = HuBERT(model_hub, save_path,output_all_hiddens=True)
# model = DiscreteSSL(save_path, ssl_model, kmeans_repo_id=kmeans_repo_id, kmeans_dataset=kmeans_dataset,num_clusters=num_clusters)
model_hub = "fnlp/SpeechTokenizer"
save_path = "savedir"
model =Tokenizer_SpeechTokenizer(model_hub, save_path)  # doctest: +SKIP
tokenizer= Tokenizer(model)
audio = torch.randn(4, 1000)
length = torch.tensor([1.0, .5, .75, 1.0])
tokens = tokenizer.encode(audio, length,num_codebooks=2)
print(tokens.shape)
rec = tokenizer.decode(tokens)
print(rec.shape)