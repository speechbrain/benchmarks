"""
Unified interface for tokenizers, standardizing the output shape of encode and decode functions.

This class reshapes the outputs of various tokenizers to ensure consistency, simplifying integration with recipes and workflows.

Authors
---------
* Pooneh Mousavi, 2024
"""
import sys
import os
import torch
from abc import ABC, abstractmethod
from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import (
    DiscreteSSL,
)
from speechbrain.lobes.models.discrete.dac import DAC
from speechbrain.lobes.models.discrete.speechtokenizer import SpeechTokenizer
from speechbrain.lobes.models.discrete.wavtokenizer import WavTokenizer
from speechbrain.lobes.models.huggingface_transformers.mimi import Mimi

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # noqa: E402
sys.path.append(base_dir)  # noqa: E402

from model.sq_codec import SQCodec  # noqa: E402


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers that encode signals into discrete tokens
    and decode tokens back into signals.

    This class defines the essential methods that any tokenizer must implement,
    including encoding, decoding, and retrieving pretrained embeddings.

    Naming Convenstion
    ------------------
    B : int
        Batch size.
    T : int
        Sequence length in the time domain.
    N : int
        Sequence length in the token domain.
    C : int
        Vocabulary size, assuming each codebook has the same number of tokens.
    K : int
        Number of codebooks.
    """

    def __init__(self):
        """
        Initialize the BaseTokenizer.

        This is a base constructor that other tokenizers can extend.
        """
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
        """
        Encode a signal into discrete tokens.

        Arguments
        ---------
        signal : torch.Tensor
            Input signal with shape [B, T].
        lengths : torch.Tensor
            Lengths of each sequence in the batch, with shape [B].
        num_codebooks : int, optional
            Number of codebooks to use for encoding. If None, all codebooks are used (default: None).
            If specified as an int, the tokens will be truncated to include only the first `num_codebooks` codebooks. If specified as a list,
            the tokens will include only the codebooks at the specified indices.
        **kwargs : dict
            Additional arguments for the tokenizer.

        Returns
        -------
        tokens : torch.Tensor
            Discretized tokens with shape [B, N, K].
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        """
        Decode discrete tokens back into a signal.

        Arguments
        ---------
        tokens : torch.Tensor
            Input tokens with shape [B, N, K].
        **kwargs : dict
            Additional arguments for the tokenizer.

        Returns
        -------
        signal : torch.Tensor
            Reconstructed signal with shape [B, T].
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def get_pretrained_embeddings(self, vocab_size, num_codebooks, **kwargs):
        """
        Retrieve pretrained embeddings for the tokenizer.

        Arguments
        ---------
        vocab_size : int
            Number of tokens in each codebook.
        num_codebooks : int
            Number of codebooks.
        **kwargs : dict
            Additional arguments for embedding retrieval.

        Returns
        -------
        embeddings : torch.Tensor
            Pretrained embedding weights with shape [K * C, H], where H is the embedding dimension.
        """
        pass


class EncodecTokenizer(Encodec, BaseTokenizer):
    """This is a wrapper for the Encodec implemented in the SpeechBrain main repository.

    Source paper:
        https://arxiv.org/abs/2210.13438
    Example
    -------
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = EncodecTokenizer(model_hub, save_path)
    >>> emb=model.get_pretrained_embeddings()
    >>> emb.shape
    torch.Size([2048, 128])
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens= model.sig_to_tokens(audio, length)
    >>> tokens.shape
    torch.Size([4, 4, 2])
    >>> rec = model.tokens_to_sig(tokens, lenght=length)
    >>> rec.shape
    torch.Size([4, 1280]
    """

    def __init__(self, *args, **kwargs):
        Encodec.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
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
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        embeddings = self.vocabulary
        return embeddings.reshape(-1, embeddings.shape[-1])


class DACTokenizer(DAC, BaseTokenizer):
    """This is a wrapper for the DAC implemented in the SpeechBrain main repository.

    Source paper:
        http://arxiv.org/abs/2306.06546
    Example
    -------
    >>> model = DACTokenizer(load_pretrained=True, model_type="24KHz", model_bitrate="8kbps", tag="latest")
    >>> audio = torch.randn(4, 16000)
    >>> emb=model.get_pretrained_embeddings(vocab_size=1024, num_codebooks=8)
    >>> emb.shape
    torch.Size([8192, 1024])
    >>> tokens= model.sig_to_tokens(audio)
    >>> tokens.shape
    torch.Size([4, 50, 32])
    >>> rec = model.tokens_to_sig(tokens)
    >>> rec.shape
    torch.Size([4, 15992])
    """

    def __init__(self, *args, **kwargs):
        DAC.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
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
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        toks = torch.arange(vocab_size).to(next(self.parameters()).device)
        toks = toks[:, None, None].expand(-1, num_codebooks, -1).clone()
        self.eval()
        z_q, z_p, _ = self.quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)
        z_qs = [
            self.quantizer.quantizers[i].out_proj(z_p_i)
            for i, z_p_i in enumerate(z_ps)
        ]
        return torch.cat(z_qs)[:, :, 0]


class SpeechTokenizerWrapper(SpeechTokenizer, BaseTokenizer):
    """This is a wrapper for the SpeechTokenizer implemented in the SpeechBrain main repository.

    Source paper:
        https://arxiv.org/abs/2308.16692
    Example
    -------
    >>> audio = torch.rand([10, 600])
    >>> model_hub = "fnlp/SpeechTokenizer"
    >>> save_path = "savedir"
    >>> model = SpeechTokenizerWrapper(model_hub, save_path)
    >>> emb=model.get_pretrained_embeddings(vocab_size=1024, num_codebooks=8)
    >>> emb.shape
    torch.Size([8192, 1024])
    >>> tokens= model.sig_to_tokens(audio)
    >>> tokens.shape
    torch.Size([10, 2, 8])
    >>> rec = model.tokens_to_sig(tokens)
    >>> rec.shape
    torch.Size([10, 640])
    """

    def __init__(self, *args, **kwargs):
        SpeechTokenizer.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
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
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        toks = torch.arange(vocab_size).to(next(self.parameters()).device)
        toks = toks[None, :, None].expand(num_codebooks, -1, -1).clone()
        self.eval()
        embs = [
            self.model.quantizer.vq.layers[i].decode(indices)
            for i, indices in enumerate(toks)
        ]
        return torch.cat(embs)[:, :, 0]


class DiscreteSSLTokenizer(DiscreteSSL, BaseTokenizer):
    """This is a wrapper for the Encodec implemented in the SpeechBrain main repository.

    Source paper:
        https://arxiv.org/abs/2210.13438
    Example
    -------
    >>> from speechbrain.lobes.models.huggingface_transformers.wavlm import (WavLM)
    >>> inputs = torch.rand([3, 2000])
    >>> model_hub = "microsoft/wavlm-large"
    >>> save_path = "savedir"
    >>> ssl_layer_num = [7,23]
    >>> deduplicate =[False, True]
    >>> bpe_tokenizers=[None, None]
    >>> vocoder_repo_id = "speechbrain/hifigan-wavlm-k1000-LibriTTS"
    >>> kmeans_dataset = "LibriSpeech"
    >>> num_clusters = 1000
    >>> ssl_model = WavLM(model_hub, save_path,output_all_hiddens=True)
    >>> model = DiscreteSSLTokenizer(save_path, ssl_model, vocoder_repo_id=vocoder_repo_id, kmeans_dataset=kmeans_dataset,num_clusters=num_clusters)
    >>> emb=model.get_pretrained_embeddings(num_codebooks=ssl_layer_num)
    >>> emb.shape
    torch.Size([2000, 1024])
    >>> tokens= model.sig_to_tokens(inputs,num_codebooks=ssl_layer_num, deduplicates=deduplicate, bpe_tokenizers=bpe_tokenizers)
    >>> tokens.shape
    torch.Size([3, 6, 2])
    >>> sig = model.tokens_to_sig(tokens, SSL_layers=ssl_layer_num)
    >>> sig.shape
    torch.Size([3, 1920])
    """

    def __init__(self, *args, **kwargs):
        DiscreteSSL.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _, _ = self.encode(
            signal, lengths, SSL_layers=num_codebooks, **kwargs
        )
        return tokens

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        return self.decode(tokens, **kwargs).squeeze(1)

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        embs = []
        for layer_num, vocabulary in zip(
            self.ssl_layer_ids, self.vocabularies,
        ):
            if layer_num not in num_codebooks:
                continue
            embs.append(torch.as_tensor(vocabulary, dtype=torch.float32))
        embs = torch.cat(embs)
        return embs


class MimiTokenizer(Mimi, BaseTokenizer):
    """This is a wrapper for the Mimi implemented in the SpeechBrain main repository.

    Source paper:
        https://kyutai.org/Moshi.pdf
    Example
    -------
    >>> model_hub = "kyutai/mimi"
    >>> save_path = "savedir"
    >>> model = MimiTokenizer(model_hub, save_path)
    >>> emb=model.get_pretrained_embeddings()
    >>> emb.shape
    torch.Size([16384, 256])
    >>> audio = torch.randn(4, 48000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens = model.sig_to_tokens(audio, length)
    >>> tokens.shape
    torch.Size([4, 25, 8])
    >>> rec = model.tokens_to_sig(tokens, length=length)
    >>> rec.shape
    torch.Size([4, 48000])
    """

    def __init__(self, *args, **kwargs):
        Mimi.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self.encode(signal, lengths)
        if num_codebooks:
            if tokens.shape[1] < num_codebooks:
                raise ValueError(
                    f"Model only outputs {tokens.shape[1]} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[:, :num_codebooks, :]
        return tokens.movedim(-1, -2)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        signal = self.decode(tokens.movedim(-1, -2), **kwargs)[:, 0]
        return signal

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        return self.embeddings.view(-1, self.embeddings.size(-1))


class WavTokenizerWrapper(WavTokenizer, BaseTokenizer):
    """This is a wrapper for the WavTokenizer implemented in the SpeechBrain main repository.

    Source paper:
        https://arxiv.org/abs/2408.16532

    Example
    -------
    >>> model_hub = "novateur/WavTokenizer"
    >>> save_path = "savedir"
    >>> config="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    >>> checkpoint="WavTokenizer_small_600_24k_4096.ckpt"
    >>> model = WavTokenizerWrapper(model_hub, save_path,config=config,checkpoint=checkpoint)
    >>> emb=model.get_pretrained_embeddings()
    >>> emb.shape
    torch.Size([4096, 512])
    >>> audio = torch.randn(4, 48000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens= model.sig_to_tokens(audio, length)
    >>> tokens.shape
    torch.Size([4, 80, 1])
    >>> rec = model.tokens_to_sig(tokens)
    >>> rec.shape
    torch.Size([4, 48000])
    """

    def __init__(self, *args, **kwargs):
        WavTokenizer.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self.encode(signal)
        if num_codebooks:
            if tokens.shape[1] < num_codebooks:
                raise ValueError(
                    f"Model only outputs {tokens.shape[1]} codebooks, but {num_codebooks} requested"
                )
            tokens = tokens[:, :num_codebooks, :]

        return tokens.movedim(-2, -1)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        signal = self.decode(tokens.movedim(-1, -2))
        return signal

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        return self.embeddings


class SQCodecTokenizer(SQCodec, BaseTokenizer):
    """This is a wrapper for the SQCoced implemented in the model folder.

    Source paper:
        https://arxiv.org/abs/2406.02328, https://arxiv.org/abs/2408.13893


    Make sure that you download and extract the SQ-codec.zip in save_path from following Huggingface repo:
        - HF repo: https://huggingface.co/Dongchao/UniAudio/blob/main/SQ-Codec.zip

    Example
    -------
    >>> save_path = "savedir"
    >>> config = "config.yaml"
    >>> checkpoint = "ckpt_00190000.pth"
    >>> model = SQCodecTokenizer(save_path, config, checkpoint)
    >>> audio = torch.randn(3, 48000)
    >>> tokens = model.sig_to_tokens(audio)
    >>> tokens.shape
    torch.Size([3, 150, 4])
    >>> rec = model.tokens_to_sig(tokens)
    >>> rec.shape
    torch.Size([3, 48000]
    """

    def __init__(self, *args, **kwargs):
        SQCodec.__init__(self, *args, **kwargs)
        BaseTokenizer.__init__(self)

    @torch.no_grad()
    def sig_to_tokens(self, signal, lengths=None, num_codebooks=None, **kwargs):
        self.eval()
        tokens, _ = self.encode(signal)
        return tokens.view(tokens.shape[0], -1, self.n_codebook)

    @torch.no_grad()
    def tokens_to_sig(self, tokens, **kwargs):
        self.eval()
        signal = self.decode(tokens.view(tokens.shape[0], -1), **kwargs)
        return signal.squeeze(1)

    @torch.no_grad()
    def get_pretrained_embeddings(
        self, vocab_size=None, num_codebooks=None, **kwargs
    ):
        """
        This method is not implemented for SQCodec, as it uses scalar quantization
        and does not have any trainable quantizer or embedding.
        """
        raise ValueError(
            "SQCodec does not have any trainable quantizer or embedding since it uses scalar quantization."
        )
