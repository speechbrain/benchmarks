"""An adaptation of ESPNET VALL-E
Originally by Jinchuan Tian

https://github.com/espnet/espnet

Authors
 * Artem Ploujnikov 2024 (adaptation only)
"""

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Vall-E: https://arxiv.org/abs/2301.02111

import logging
import torch
from typing import Tuple, Optional
from speechbrain.dataio.dataio import length_to_mask

from torch import Tensor
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

from speechbrain.nnet.losses import reduce_loss
from speechbrain.nnet.losses import truncate


@dataclass
class SpeechLMInferenceOptions:
    """Inference options
    """

    device: str = None
    search_algo: str = "topk_sampling"
    nbest: int = 1
    sampling_temperature: float = 1.0
    top_k: int = 20
    maxlenratio: float = 0.0
    minlenratio: float = 0.0
    eos: int = 5
    start: int = 1
    masks: torch.Tensor = None
    nq: int = None
    allow_invalid: bool = True


class ValleLM(nn.Module):
    """The Vall-E TTS model (decoder-only transformer), adopted from
    ESPNET2

    Arguments
    ---------
    vocab_size : int
        Dimention of vocabulary.
    nq : int
        Number of codes for each token / frame, usually for speech codec.
    share_emb : bool
        If true, share the embedding and lm_head weight.
    qk_norm : bool
        If true, apply LayerNorm to q and k in atention.
    dropout : float
        dropout rate for attention layers.
    att_unit: int
        Dimention of Transformer attention.
    head : int
        Number of heads in Transformer attention.
    ar_layer : int
        Number of layers in AR Transformer.
    nar_layer : int
        Number of layers in NAR Transformer.
    n_ctx : int
        maximum context length of AR & NAR Transformer.
    """

    def __init__(
        self,
        vocab_size,
        nq,
        pad_id=0,
        share_emb=True,
        qk_norm=False,
        dropout=0.0,
        att_unit=256,
        head=2,
        ar_layer=4,
        nar_layer=4,
        n_ctx=3000,
    ):
        super().__init__()

        self.emb = torch.nn.Embedding(vocab_size, att_unit)
        self.lm_head = torch.nn.Linear(att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        self.ar_decoder = TransformerDecoder(
            n_ctx=n_ctx,
            n_state=att_unit,
            n_head=head,
            n_layer=ar_layer,
            qk_norm=qk_norm,
            dropout=dropout,
        )
        if nq > 1:
            # NOTE: An NAR encoder is not needed if there is only one track
            self.nar_decoder = ValleNARDecoder(
                n_level=nq - 1,
                n_ctx=n_ctx,
                n_state=att_unit,
                n_head=head,
                n_layer=nar_layer,
                qk_norm=qk_norm,
                dropout=dropout,
            )

        self.nq = nq
        self.n_ctx = n_ctx
        self.pad_id = pad_id
        self._initialize()

    def forward(
        self,
        dec_seq,
        dec_seq_lengths=None,
        prefix_len=None,
        conti_feats=None,
        nar_level_idx=1,
        predict_ar=True,
        predict_nar=True,
    ):
        """Vall-E forward for training

        Arguments
        ---------
        dec_seq : torch.Tensor
            Batch of decoder sequences (B, T, nq).
        dec_seq_lengths : torch.Tensor
            Lengths of batched decoder sequences (B,).
        enc_seq : torch.Tensor
            Batch of encoder sequences (B, T, nq), keep
            the interface, may not be used.
        enc_seq_lengths : torch.Tensor
            Lengths of batched encoder sequences (B,),
            keep the interface, may not be used.
        prefix_len : torch.Tensor
            Lengths of condition part in dec_seq (B,).
        nar_level_idx : int
            the index of the non-autoregressive level to train
        predict_ar : bool
            Whether to make an autoregressive prediction
        predict_nar : bool
            Whether to make a non-autoregressive prediction

        Returns
        -------
        logits_ar : torch.Tensor
            Autoregressive predictions
        logits_nar : torch.Tensor
            Non-autoregressive predictions
        """

        assert dec_seq.dim() == 3

        dec_seq_emb = self.emb(dec_seq)  # [B, T, nq, D]
        dec_seq_emb, _ = install_continuous_features(
            dec_seq_emb, None, conti_feats
        )

        # Auto-Regressive part
        if predict_ar:
            input_ar_emb = self.prepare_input(dec_seq_emb, prefix_len, 1)[
                :, :-1
            ]  # [B, T, D]
            h_ar = self.ar_decoder(input_ar_emb)

        # Non-Auto-Regressive part
        if predict_nar:
            input_nar_emb = self.prepare_input(
                dec_seq_emb, prefix_len, nar_level_idx
            )[
                :, 1:
            ]  # [B, T, V]
            max_len = dec_seq.size(1)
            mask = length_to_mask(dec_seq_lengths * max_len - 1, max_len - 1).bool()
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
            h_nar = self.nar_decoder(input_nar_emb, nar_level_idx - 1, mask=mask)

        # Logits
        logits_ar, logits_nar = None, None
        if predict_ar:
            logits_ar = self.lm_head(h_ar)
        if predict_nar:
            logits_nar = self.lm_head(h_nar)

        return logits_ar, logits_nar

    def prepare_input(self, dec_seq_emb, prefix_len, level):
        # NOTE(Jinchuan): have to use "expand" here but maybe lead to extra memory usage.
        # This is because both prefix_mask and level_mask are broadcastable and will
        # trigger user warning.

        # (1) level mask, [B, 1, nq, 1], True is to include
        if isinstance(level, int):
            level = torch.ones_like(dec_seq_emb[:, 0, 0, 0]) * level
        level_mask = length_to_mask(level, self.nq).bool()
        level_mask = (
            level_mask.unsqueeze(1).unsqueeze(3).expand(dec_seq_emb.size())
        )

        # (2) prefix mask, [B, T, 1, 1], True is the prefix
        prefix_mask = length_to_mask(
            prefix_len * dec_seq_emb.size(1), dec_seq_emb.size(1)
        ).bool()
        prefix_mask = (
            prefix_mask.unsqueeze(2).unsqueeze(3).expand(dec_seq_emb.size())
        )

        # (3) mask and then sum in nq-axis.
        mask = torch.logical_or(level_mask, prefix_mask)
        return dec_seq_emb.masked_fill(~mask, 0.0).sum(2)

    @torch.no_grad()
    def inference(
        self, prefix, opts, enc_seq=None, suffix=None,
    ):
        """Vall-E Inference.

        Arguments
        ---------
        prefix : torch.Tensor
            Prefix part of dec_seq (B, T, nq).
        opts : SpeechLMInferenceOptions
            inference options.
        enc_seq : torch.Tensor
            Encoder token sequence (B, T, nq).
        suffix : torch.Tensor
            suffix part of dec_seq (B, T, nq),
            usually the target sequence for teacher-forcing.

        Returns
        -------
        gen_tokens_list : list
            Generated tokens
        gen_scores_list : list
            The scores associated with the generated tokens
        """

        # (1) initialization
        cache = self.ar_decoder.init()

        # (2) auto-regressive prefix forward on first code layer
        prefix = prefix.expand(opts.nbest, -1, -1)
        if opts.search_algo == "teacher_force":
            suffix = suffix.expand(opts.nbest, -1, -1)
        prefix_emb = self.emb(prefix).sum(dim=2)  # [B, T, D]
        _ = self.ar_decoder(prefix_emb, kv_cache=cache)

        # (3) auto-regressive loop on first code layer
        # (3.1) AR initialization
        minlen = (
            int(prefix.size(1) * opts.minlenratio)
            if opts.minlenratio > 0
            else 0
        )
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force":
            assert suffix is not None
            minlen = suffix.size(1)
            maxlen = suffix.size(1)
        if maxlen + prefix.size(1) > self.n_ctx:
            maxlen = self.n_ctx - prefix.size(1)
        logging.info(f"maxlen={maxlen}, minlen={minlen}")

        generated = {"token": [], "score": []}
        finish_idx = (
            torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        )
        prev_tok = (
            torch.Tensor([opts.start])
            .tile(opts.nbest, 1)
            .long()
            .to(opts.device)
        )
        modality_index = prev_tok.flatten()
        mask = modality_index_to_mask(modality_index, opts)
        mask_cache = []
        modality_tokens = torch.tensor(
            list(opts.masks.keys()), device=prefix.device
        )

        for step in range(maxlen):
            #  (3.2) AR loop
            prev_emb = self.emb(prev_tok)  # [B, 1, D]
            h_ar = self.ar_decoder(prev_emb, kv_cache=cache)
            logits = self.lm_head(h_ar)  # [B, 1, V]
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                mask,
                allow_eos=step >= minlen,
                nq_level=0,
            )
            # [B, 1, 1] -> [B, 1]
            gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step : step + 1, 0]
            else:
                prev_tok = gen_tok  # [B, 1]

            # (3.3) detect modality swtich
            mask_cache.append(mask.clone())
            modality_change_mask = torch.isin(prev_tok[:, 0], modality_tokens)
            # Note: The ESPNET VALL-E had
            # modality_change_mask = torch.logical_and(
            #    prev_tok[:, 0] >= 32, prev_tok[:, 0] < 64,
            # )
            if torch.any(modality_change_mask):
                modality_index = torch.where(
                    modality_change_mask, prev_tok[:, 0], modality_index,
                )
                mask = modality_index_to_mask(modality_index, opts)
                logging.warning(
                    f"Step {step}: change modality index {modality_index}"
                )

            # (3.4) detect ended hypotheses.
            finish_idx = torch.where(
                torch.logical_and(prev_tok[:, 0] == opts.eos, finish_idx == -1),
                step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                break

            if step == maxlen - 1:
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx}"
                    f"Consider increasing the maxlenratio"
                )

        logging.info(f"Terminate at steps: {finish_idx.cpu().tolist()}")

        # (3.4) finalize auto-regressive
        if opts.allow_invalid:
            valid_idx = torch.arange(len(finish_idx), device=finish_idx.device)
            finish_idx = torch.where(finish_idx == -1, step, finish_idx)
        else:
            valid_idx = finish_idx.ne(-1).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            self.ar_decoder.reset()
            logging.warning(f"No valid examples. Return None")
            return [], []
        elif len(valid_idx) < prefix.size(0):
            logging.info(f"Only {len(valid_idx)} of {prefix.size(0)} are valid")

        finish_idx = finish_idx[valid_idx]
        prefix_emb = prefix_emb[valid_idx]
        if opts.search_algo == "teacher_force":
            suffix = suffix[valid_idx]
        gen_tokens_ar = torch.cat(generated["token"], dim=1)[
            valid_idx
        ].unsqueeze(
            2
        )  # [B, T, 1]
        gen_scores_ar = torch.cat(generated["score"], dim=1)[
            valid_idx
        ].unsqueeze(2)
        gen_tokens_ar = gen_tokens_ar[:, : finish_idx.max() + 1]  # idx -> count
        gen_scores_ar = gen_scores_ar[:, : finish_idx.max() + 1]

        self.ar_decoder.reset()

        # (4) non-auto-regressive loop on the remained code layers
        # (4.1) NAR initialization
        if opts.search_algo == "teacher_force":
            prev_tok = suffix[:, :, 0]
        else:
            prev_tok = gen_tokens_ar[:, :, 0]
        start_emb = self.emb.weight[opts.start].tile(
            len(valid_idx), 1, 1
        )  # [B, 1, D]
        prev_emb = torch.cat(
            [prefix_emb[:, 1:], start_emb, self.emb(prev_tok)], dim=1
        )  # [B, T, D]

        ones = torch.ones_like(valid_idx)
        mask = length_to_mask(prefix.size(1) + finish_idx + 1).bool()
        mask = mask.unsqueeze(1).unsqueeze(1)
        generated = {"token": [], "score": []}

        mask_cache = [mask_cache[0]] * prefix.size(1) + mask_cache
        vocab_mask = torch.cat(mask_cache, dim=1)

        # (4.2) NAR loop
        if self.nq > 1:
            for step in range(1, opts.nq):
                h_nar = self.nar_decoder(
                    prev_emb, ones * step - 1, mask=mask
                )  # [B, T, D]
                logits = self.lm_head(h_nar)
                gen_tok, gen_score = logits_to_tokens(
                    logits.unsqueeze(2),
                    opts,
                    vocab_mask,
                    search_algo="greedy_search",
                    allow_eos=False,
                    nq_level=step,
                )
                gen_tok, gen_score = (
                    gen_tok.squeeze(2),
                    gen_score.squeeze(2),
                )  # [B, T]

                generated["token"].append(gen_tok[:, prefix.size(1) :])
                generated["score"].append(gen_score[:, prefix.size(1) :])

                if opts.search_algo == "teacher_force":
                    prev_tok = suffix[:, :, step]
                else:
                    prev_tok = generated["token"][-1]
                prev_emb[:, prefix.size(1) :] += self.emb(prev_tok)  # [B, T, D]
                prev_emb[:, prefix.size(1) - 1 : prefix.size(1)] += start_emb

            # (5) combine AR and NAR results
            gen_tokens_nar = torch.stack(generated["token"], dim=2)  # [B, T, nq]
            gen_scores_nar = torch.stack(generated["score"], dim=2)

            gen_tokens = torch.cat(
                [gen_tokens_ar, gen_tokens_nar], dim=2
            )  # [B, T, nq]
            gen_scores = torch.cat([gen_scores_ar, gen_scores_nar], dim=2)
        else:
            gen_tokens = gen_tokens_ar
            gen_scores = gen_scores_ar

        gen_tokens_list, gen_scores_list = [], []
        for b in range(len(valid_idx)):
            gen_tokens_list.append(gen_tokens[b][: finish_idx[b]])
            gen_scores_list.append(gen_scores[b][: finish_idx[b]])

        return gen_tokens_list, gen_scores_list

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)


class ResidualAttentionBlock(nn.Module):
    """A VALL-E residual attention block

    Arguments
    ---------
    n_state : int
        The number of states
    n_head : int
        The number of heads
    cross_attention : bool
        Whether to use cross-attention
    causal : bool
        Whether to operate in causal mode (i.e. avoid attending
        to future steps)
    qk_norm : bool
        Whether to normalize queries and keys
    dropout : float
        The dropout probability
    """

    def __init__(
        self,
        n_state,
        n_head,
        cross_attention=False,
        causal=False,
        qk_norm=False,
        dropout=0.0,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            n_state, n_head, causal=causal, qk_norm=qk_norm, dropout=dropout,
        )
        self.attn_ln = LayerNorm(n_state)
        self.attn_dropout = nn.Dropout(p=dropout)

        self.cross_attn = (
            MultiHeadAttention(
                n_state, n_head, causal=False, qk_norm=qk_norm, dropout=dropout,
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.cross_attn_dropout = (
            nn.Dropout(p=dropout) if cross_attention else None
        )

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.mlp_dropout = nn.Dropout(p=dropout)

    def forward(
        self, x, xa=None, mask=None, kv_cache=None,
    ):
        """The forward pass implementation

        Arguments
        ---------
        x : torch.Tensor
            the feature tensor
        xa : torch.Tensor
            The tensor for cross-attention
        mask : torch.Tensor
            The attention mask to be applied

        """
        x = x + self.attn_dropout(
            self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        )
        if self.cross_attn:
            x = x + self.cross_attn_dropout(
                self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
            )
        x = x + self.mlp_dropout(self.mlp(self.mlp_ln(x)))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_ctx,
        n_state,
        n_head,
        n_layer,
        causal=True,
        qk_norm=False,
        dropout=0.0,
        layer_class=ResidualAttentionBlock,
    ):
        """A custom transformer decoder implementation for VALL-E

        Arguments
        ---------
        n_ctx : int
            The context length
        n_state : int
            The number of states
        n_head : int
            The number of heads
        n_layer : int
            The number of layers
        causal : bool
            Whether to operate in causal mode (i.e. avoid attending
            to future steps)
        qk_norm : bool
            Whether to normalize queries and keys
        dropout : float
            The dropout probability
        layer_class : type
            The layer type to be used
        """
        super().__init__()

        self.pos_emb = nn.Embedding(n_ctx, n_state)

        self.blocks = nn.ModuleList(
            [
                layer_class(
                    n_state=n_state,
                    n_head=n_head,
                    cross_attention=False,
                    causal=causal,
                    qk_norm=qk_norm,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        self.causal = causal
        self.kv_cache = None

    def forward(
        self, x, mask=None, kv_cache=None,
    ):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            the feature tensor
        mask : torch.Tensor
            The attention mask to be applied
        kv_cache : dict
            The key/value cache (for inference)

        Returns
        -------
        result : torch.Tensor
            The decoder output
        """
        if self.causal and mask is not None:
            raise ValueError("Causal Transformer dones't allow mask")

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=kv_cache)

        x = self.ln(x)
        return x

    def init(self):
        """Initializes the key/value cache and the hooks to update it"""
        self.kv_cache, self.hooks = install_kv_cache_hook(self, self.kv_cache)
        return self.kv_cache

    def reset(self):
        """Resets the key-value cache"""
        for hook in self.hooks:
            hook.remove()
        self.kv_cache = None


class LayerNorm(nn.LayerNorm):
    """A layer normalziation wrapper"""

    def forward(self, x):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            The tensor to be normalized

        Returns
        -------
        result : torch.Tensor
            A normalzied tensor
        """
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class ResidualAttentionBlockAdaLN(ResidualAttentionBlock):
    """"The Vall-E Adaptive Residual Attention Block

    Arguments
    ---------
    n_state : int
        The number of states
    n_head : int
        The number of states
    n_head : int
        The number of attention heads
    cross_attention : bool
        The number of attention heads
    causal : bool
        Whether to operate in causal mode (i.e. avoid attending
        to future steps)
    qk_norm : bool
        Queries/Keys Normalization
    dropout : float
        The dropout probability
    """

    def __init__(
        self,
        n_state,
        n_head,
        cross_attention=False,
        causal=False,
        qk_norm=False,
        dropout=0.0,
    ):
        super(ResidualAttentionBlockAdaLN, self).__init__(
            n_state=n_state,
            n_head=n_head,
            cross_attention=cross_attention,
            causal=causal,
            qk_norm=qk_norm,
            dropout=dropout,
        )

        self.attn_ln = AdaLN(n_state)
        self.mlp_ln = AdaLN(n_state)

    def forward(
        self, x, level, xa=None, mask=None, kv_cache=None,
    ):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            The source tensor
        level : torch.Tensor
            The level numbers for each batch element
        xa : torch.Tensor
            The sequence for cross attention
        mask : torch.Tensor
            The attention mask
        kv_cache : dict
            The key/value cache (for inference)
        """
        x = x + self.attn_dropout(
            self.attn(self.attn_ln(x, level), mask=mask, kv_cache=kv_cache)
        )
        if self.cross_attn:
            x = x + self.cross_attn_dropout(
                self.cross_attn(
                    self.cross_attn_ln(x, level), xa, kv_cache=kv_cache
                )
            )
        x = x + self.mlp_dropout(self.mlp(self.mlp_ln(x, level)))
        return x


class ValleNARDecoder(TransformerDecoder):
    def __init__(
        self,
        n_level,
        n_ctx,
        n_state,
        n_head,
        n_layer,
        causal=False,
        qk_norm=False,
        dropout=0.0,
        layer_class=ResidualAttentionBlockAdaLN,
    ):
        """The VALL-E non-autoregressive decoder

        Arguments
        ---------
        n_level : int
            The number of levels
        n_ctx : int
            The context length
        n_state : int
            The number of states
        n_head : int
            The number of attention heads
        n_layer : int
            The number of layers
        causal : bool
            Whether to operate in causal mode (i.e. avoid attending
            to future steps)
        qk_norm : bool
            Queries/Keys Normalization
        dropout : float
            The dropout probability
        layer_class : type
            The layer class to use
        """
        super().__init__(
            n_ctx=n_ctx,
            n_state=n_state,
            n_head=n_head,
            n_layer=n_layer,
            causal=causal,
            qk_norm=qk_norm,
            dropout=dropout,
            layer_class=layer_class,
        )

        self.level_emb = nn.Embedding(n_level, n_state)
        self.ln = AdaLN(n_state)

    def forward(
        self, x, level, mask=None, kv_cache=None,
    ):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            The source tensor
        level : torch.Tensor
            The level numbers for each batch element
        mask : torch.Tensor
            The attention mask
        kv_cache : dict
            The key/value cache (for inference)
        """
        if self.causal and mask is not None:
            raise ValueError("mask is not allowed when causal")

        level = self.level_emb(level)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, level=level, mask=mask, kv_cache=kv_cache)

        x = self.ln(x, level)
        return x


class MultiHeadAttention(nn.Module):
    """A Multi-Head Attention implementation

    Arguments
    ---------
    n_state : int
        The number of states
    n_head : int
        The number of attention heads
    causal : bool
        Whether to operate in causal mode (i.e. avoid attending
        to future steps)
    qk_norm : bool
        Queries/Keys Normalization
    dropout : float
        The dropout probability
    """

    def __init__(
        self, n_state, n_head, causal=False, qk_norm=False, dropout=0.0,
    ):
        super().__init__()
        assert n_state % n_head == 0
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.causal = causal
        self.dropout = dropout

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LayerNorm(n_state // n_head)
            self.k_norm = LayerNorm(n_state // n_head)

    def forward(
        self, x, xa=None, mask=None, kv_cache=None,
    ):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            The source tensor
        xa : torch.Tensor
            The sequence for cross attention
        mask : torch.Tensor
            The attention mask
        kv_cache : dict
            The key/value cache (for inference)
        """
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)

        return self.out(wv)

    def qkv_attention(self, q, k, v, mask=None):
        """Computes self-attention

        Arguments
        ---------
        q : torch.Tensor
            The queries tensor
        k : torch.Tensor
            The keys tensor
        v : torch.Tensor
            The values tensor

        Returns
        -------
        wv : torch.Tensor
            The attention output
        """
        if self.causal and mask is not None:
            raise ValueError("mask is not allowed when the attention is causal")

        if self.causal and q.size(1) == k.size(1):
            causal = True
        else:
            causal = False

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            wv = (
                F.scaled_dot_product_attention(
                    q, k, v, mask, is_causal=causal, dropout_p=self.dropout
                )
                .permute(0, 2, 1, 3)
                .flatten(start_dim=2)
            )

        return wv


class AdaLN(nn.Module):
    """Adaptive Layer Normalization, a Layer Norm implementation
    that learns an affine transformation based on the level
    embedding

    Arguemnts
    ---------
    n_state : int
        The number of states
    eps : float
        The layer norm epsilon parameter"""

    def __init__(self, n_state, eps=1e-5):
        super().__init__()
        self.weight = nn.Linear(n_state, n_state, bias=False)
        self.bias = nn.Linear(n_state, n_state, bias=False)
        nn.init.constant_(self.weight.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

        self.n_state = n_state
        self.eps = eps

    def forward(self, x, level_emb):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            The tensor
        level_emb : torch.Tensor
            The level embedding
        """
        w = self.weight(level_emb).unsqueeze(1)
        b = self.bias(level_emb).unsqueeze(1)
        x = nn.functional.layer_norm(x, (self.n_state,), eps=self.eps)
        x = w * x + b
        return x


def install_kv_cache_hook(model, cache):
    """Sets up the key/value cache hook

    Arguments
    ---------
    model : torch.nn.Module
        The model
    cache : dict
        The cache content

    Returns
    -------
    cache : torch.Tensor
        The cache dictionary (new or copied)
    hooks : torch.Tensor
        The installed hooks
    """
    cache = {**cache} if cache is not None else {}
    hooks = []

    def save_to_cache(module, _, output):
        if module not in cache:
            # save as-is, for the first token or cross attention
            cache[module] = output
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def install_hooks(layer: torch.nn.Module):
        if isinstance(layer, MultiHeadAttention):
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks


def logits_to_tokens(
    logits, opts, mask, search_algo=None, allow_eos=True, nq_level=None,
):
    """
    Select the generated tokens and their scores based on logits prediction.

    Arguments
    ---------
    logits : torch.Tensor
        predicted logits, of size [B, T, nq, V]
    opts : SpeechLMInferenceOptions
        search options
    mask : torch.Tensor
        mask to specify valid tokens, of size [B, 1, nq, V]
    search_algo : str
        search algorithm
    allow_eos : bool
        whether to allow end-of-sentence prediction
    nq_level : int, optional
        if not None, only conpute the specified codec level nq.

    Returns
    -------
    gen_token_idx : torch.Tensor
        The token indexes
    gen_token_score : torch.Tensor
        The token scores
    """

    assert logits.dim() == 4
    search_algo = search_algo if search_algo is not None else opts.search_algo
    neg_inf = torch.finfo(logits.dtype).min

    # (1) Apply mask
    if nq_level is not None:
        mask = mask[:, :, nq_level : nq_level + 1]

    if allow_eos:
        mask = mask.clone()
        mask[:, :, 0, opts.eos] = False

    logits.masked_fill_(mask, neg_inf)

    # (2) token selection
    if search_algo in ["topk_sampling"]:
        topk_values, topk_indices = torch.topk(logits, opts.top_k, dim=-1)
        probs = torch.softmax(topk_values / opts.sampling_temperature, dim=-1)
        inner_indices = torch.multinomial(
            probs.flatten(end_dim=-2), num_samples=1
        ).view(probs[..., :1].size())
        gen_token_idx = torch.gather(topk_indices, -1, inner_indices).squeeze(
            -1
        )
        gen_token_score = (
            torch.gather(probs, -1, inner_indices).squeeze(-1).log()
        )

    elif search_algo in ["topp_sampling"]:
        probs = torch.softmax(logits / opts.sampling_temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        accum_probs = torch.cumsum(sorted_probs, dim=-1)
        clip_probs = torch.where(accum_probs <= opts.top_p, sorted_probs, 0.0)
        # always keep at least one candidate no matter what value it is
        if torch.any(clip_probs[..., 0] == 0.0):
            clip_probs[..., 0] = sorted_probs[..., 0]
        clip_probs = clip_probs / clip_probs.sum(dim=-1, keepdim=True)
        inner_indices = torch.multinomial(
            clip_probs.flatten(end_dim=-2), num_samples=1
        ).view(clip_probs[..., :1].size())
        gen_token_idx = torch.gather(sorted_indices, -1, inner_indices).squeeze(
            -1
        )
        gen_token_score = (
            torch.gather(clip_probs, -1, inner_indices).squeeze(-1).log()
        )

    elif search_algo in ["greedy_search", "teacher_force"]:
        probs = logits.softmax(dim=-1)
        topk_values, topk_indices = torch.topk(logits, 1, dim=-1)
        gen_token_idx = topk_indices[:, :, :, 0]
        gen_token_score = topk_values[:, :, :, 0].log()

    else:
        raise NotImplementedError(f"opts.search_algo={opts.search_algo}")

    return gen_token_idx, gen_token_score


@torch.no_grad()
def install_continuous_features(
    dec_emb: torch.Tensor,
    enc_emb: Optional[torch.Tensor] = None,
    conti_feats: Tuple = None,
):
    if conti_feats is None:
        return dec_emb, enc_emb

    assert dec_emb.size(0) == len(conti_feats)
    if enc_emb is not None:
        assert enc_emb.size(0) == len(conti_feats)

    for b, conti_feat in enumerate(conti_feats):
        for conti_emb, start, end, part in conti_feat:
            if part == "dec":
                assert conti_emb.size(1) == dec_emb.size(2)
                dec_emb[b, start:end] = conti_emb
            else:
                assert conti_emb.size(1) == enc_emb.size(2)
                enc_emb[b, start:end] = conti_emb

    return dec_emb, enc_emb


def modality_index_to_mask(
    modality_index: torch.Tensor, inference_opts: SpeechLMInferenceOptions,
):
    assert modality_index.dim() == 1
    modality_index = modality_index.cpu().tolist()
    mask = torch.stack(
        [inference_opts.masks[idx] for idx in modality_index], dim=0
    ).unsqueeze(
        1
    )  # [B, 1, nq, V]

    return mask


def masked_nll_loss(
    log_probabilities, targets, mask, allowed_len_diff=3, reduction="mean"
):
    """Similar to the standard nll_loss from SpeechBrain
    but applies a custom mask

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    mask : torch.Tensor
        The mask for loss calculation
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.
    """
    log_probabilities, targets = truncate(
        log_probabilities, targets, allowed_len_diff
    )
    log_probabilities = log_probabilities.transpose(1, -1)
    loss = torch.nn.functional.nll_loss(
        input=log_probabilities, target=targets.long(), reduction="none"
    )
    loss *= mask
    loss = reduce_loss(loss, mask, reduction, 0.0, log_probabilities, targets)
    return loss
