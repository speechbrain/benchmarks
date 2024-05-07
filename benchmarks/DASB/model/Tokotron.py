"""A simplistic Text-to-Speech model operating on
discrete/tokenized audio representations, available in both
Transformer and RNN flavours.

NOTE: This model does not use the standard Transformer interface
in order to make it usable as both as a full model and as a
decoder-only model

Authors
* Artem Ploujnikov, 2023
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    get_lookahead_mask,
)
from speechbrain.dataio.dataio import clean_padding_
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.losses import kldiv_loss, compute_masked_loss
from benchmarks.DASB.model.custom_model import MultiEmbedding
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.decoders.seq2seq import S2STransformerBeamSearcher
from speechbrain.utils.data_utils import concat_padded_features

from enum import Enum
from collections import namedtuple
from tqdm.auto import tqdm
from functools import partial


TokotronOutput = namedtuple(
    "TokotronOutput",
    [
        "out",
        "gate_out",
        "p_eos",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
    ],
)

TokotronDecoderOutput = namedtuple(
    "TokotronDecoderOutput",
    ["out", "gate_out", "dec_self_attn", "dec_attn", "alignments", "context"],
)

TokotronDecoderInfernceOutput = namedtuple(
    "TokotronDecoderInferenceOutput",
    [
        "audio_tokens",
        "length",
        "dec_self_attn",
        "dec_attn",
        "alignments",
        "p_eos",
    ],
)

TokotronInfernceOutput = namedtuple(
    "TokotronInferenceOutput",
    [
        "audio_tokens",
        "length",
        "wav",
        "wav_length",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
        "p_eos",
    ],
)

IGNORE_IN_STATE_DICT = {"vocoder", "compression_model"}


class EosMode(Enum):
    GATE = "gate"
    TOKEN = "token"


class DecoderMode(Enum):
    AUTOREGRESSIVE = "autoregressive"
    FORWARD = "forward"


class TokotronTransformerDecoder(nn.Module):
    """The Tokotron decoder - can be used in a standalone model or as
    a component of a larger model

    Arguments
    ---------
    num_tokens : int, optional
        the number of tokens
    tokens_per_step : int, optional
        the number of tokens to be output, per transformer time step
    d_model : int, optional
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    attention_type : str
        The type of transformer attention to be used
    num_layers: int
        The number of layers
    audio_emb : torch.nn.Module, optional
        The audio embedding to be used
    activation : torch.nn.Module, optional
        The activation function to be used
    use_tgt_padding_mask : bool, optional
        whether to use a target padding mask
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
    max_decoder_steps : int, optional
        The maximum number of decoder steps used during training
    infer_max_decoder_steps : int, optional
        The maximum number of steps during autoregressive
        decoding (defaults to max_decoder_steps)
    bos_idx : int
        The index of the BOS token
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    show_inference_progress : bool, optional
        Whether to show inference progress in the console
    """

    def __init__(
        self,
        num_tokens=1024,
        tokens_per_step=2,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        attention_type="regularMHA",
        num_layers=6,
        dropout=0.2,
        target_dropout=None,
        audio_emb=None,
        audio_emb_size=128,
        activation=nn.LeakyReLU,
        use_tgt_padding_mask=False,
        audio_emb_freeze=False,
        max_decoder_steps=1000,
        infer_max_decoder_steps=None,
        bos_idx=0,
        bos_width=1,
        gate_threshold=0.5,
        gate_offset=0,
        show_inference_progress=True,
        audio_token_shift=0,
        multihead_input=True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens_per_step = tokens_per_step
        self.dec = TransformerDecoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
        in_proj_size = audio_emb_size
        if multihead_input:
            in_proj_size *= tokens_per_step
        self.tgt_in_proj = Linear(input_size=in_proj_size, n_neurons=d_model,)
        self.out_proj = Linear(
            input_size=d_model,
            n_neurons=(num_tokens + audio_token_shift) * tokens_per_step,
        )
        self.gate = Linear(input_size=d_model, n_neurons=1)
        if audio_emb is None:
            audio_emb = MultiEmbedding(
                num_embeddings=num_tokens + audio_token_shift,
                embedding_dim=audio_emb_size,
                num_heads=tokens_per_step,
                normalized=True,
                d_model=d_model,
            )
        self.positional_encoding = PositionalEncoding(
            d_model, max_decoder_steps
        )
        if target_dropout is None:
            target_dropout = dropout
        self.target_dropout = target_dropout
        self.audio_emb = audio_emb
        self.max_decoder_steps = max_decoder_steps
        if infer_max_decoder_steps is None:
            infer_max_decoder_steps = max_decoder_steps
        self.infer_max_decoder_steps = infer_max_decoder_steps
        self.attention_type = attention_type
        self.use_tgt_padding_mask = use_tgt_padding_mask
        self.audio_emb_freeze = audio_emb_freeze
        self.bos_idx = bos_idx
        self.bos_width = bos_width
        self.gate_threshold = gate_threshold
        self.gate_offset = gate_offset
        self.show_inference_progress = show_inference_progress
        if self.audio_emb_freeze:
            for parameter in self.audio_emb.parameters():
                parameter.requires_grad_(False)
        self.audio_token_shift = audio_token_shift
        self.multihead_input = multihead_input

    def decode(
        self,
        enc_out,
        tgt,
        src_length=None,
        src_key_padding_mask=None,
        tgt_length=None,
        tgt_key_padding_mask=None,
        pos_embs_src=None,
    ):
        if src_length is not None and src_key_padding_mask is None:
            src_max_len = enc_out.size(1)
            src_key_padding_mask = length_to_mask(
                src_length * src_max_len, src_max_len
            ).logical_not()

        if (
            tgt_length is not None
            and tgt_key_padding_mask is None
            and self.use_tgt_padding_mask
        ):
            tgt_max_len = tgt.size(1)
            tgt_key_padding_mask = length_to_mask(
                tgt_length * tgt_max_len, tgt_max_len
            ).logical_not()

        audio_emb = self.audio_emb(tgt)
        if self.multihead_input:
            batch_size, audio_max_len, heads, audio_dim = audio_emb.shape
            audio_emb_combined = audio_emb.reshape(
                batch_size, audio_max_len, heads * audio_dim
            )
        else:
            audio_emb_combined = audio_emb
        tgt = self.tgt_in_proj(audio_emb_combined)
        tgt = F.dropout(tgt, self.target_dropout, training=self.training)

        tgt_mask = get_lookahead_mask(tgt)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_tgt = self.positional_encoding(tgt)
        else:
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_tgt = None
        (dec_out, dec_self_attn, dec_attn,) = self.dec(
            tgt=tgt,
            memory=enc_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_tgt,
            pos_embs_src=pos_embs_src,
        )
        return dec_out, dec_self_attn, dec_attn

    def forward(
        self,
        enc_out,
        tgt,
        src_length=None,
        src_key_padding_mask=None,
        tgt_length=None,
        tgt_key_padding_mask=None,
        pos_embs_src=None,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        src : torch.Tensor
            Raw encoder outputs
        tgt : torch.Tensor
            Targets (audio tokens)
        src_length : torch.Tensor
            The relative lengths of the source sequence
        tgt_length : torch.Tensor
            Target lengths
        pos_embs_src : dict
            Source positional embeddings
        """
        tgt_shift = torch.zeros((1, tgt.size(1), 1), device=tgt.device)
        tgt_shift[:, self.bos_width :, :] += self.audio_token_shift
        dec_out, dec_self_attn, dec_attn = self.decode(
            enc_out,
            tgt + tgt_shift,
            src_length,
            src_key_padding_mask,
            tgt_length,
            tgt_key_padding_mask,
            pos_embs_src,
        )
        lin_out = self.out_proj(dec_out)
        batch_size, audio_max_len, num_tokens = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size,
            audio_max_len,
            self.tokens_per_step,
            num_tokens // self.tokens_per_step,
        )
        gate_out = self.gate(dec_out).squeeze(-1)
        return TokotronDecoderOutput(
            lin_out_heads,
            gate_out,
            dec_self_attn,
            dec_attn,
            get_alignments(dec_attn),
            {},
        )

    def init_audio_emb(self, emb):
        """Initializes audio embeddings with the specified embedding tensor - useful for re-using the
        embeddings from a pre-trained model

        Arguments
        ---------
        emb : torch.Tensor
            The embedding tensor with which to initialize
        """
        self.audio_emb.initialize(emb)


class TokotronTransformerAutoregressiveInference(nn.Module):
    """A greedy autoregressive inference implementation

    Arguments
    ---------
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    bos_idx : int, optional
        the Beginning-of-Sequence index
    max_steps : int, optional
        The maximum number of decoder steps used during training
    audio_token_shift : int, optional
        The number by which token indices will be shifted (used to introduce
        additional tokens)
    """

    def __init__(
        self,
        gate_offset,
        gate_threshold,
        tokens_per_step,
        bos_idx,
        max_steps,
        audio_token_shift,
        show_inference_progress=True,
    ):
        super().__init__()
        self.decoder = None
        self.gate_offset = gate_offset
        self.gate_threshold = gate_threshold
        self.tokens_per_step = tokens_per_step
        self.bos_idx = bos_idx
        self.max_steps = max_steps
        self.audio_token_shift = audio_token_shift
        self.show_inference_progress = show_inference_progress

    def bind(self, model):
        """Binds this inference implementation to a model

        Arguments
        ---------
        model : TokotronTransformerModel
            The transformer model
        """
        self.decoder = model.decoder

    def forward(self, enc_out, length):
        """Performs autoregressive inference

        Arguments
        ---------
        decoder : callable
            The decoder module

        enc_out : torch.Tensor
            Raw encoder outputs

        length : torch.Tensor
            Relative lengths

        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
        dec_self_attn : torch.Tensor
            Decoder self-attentions
        dec_attn : torch.Tensor
            Decoder multihead attentions (or equivalent)
        """
        with torch.no_grad():
            gate_offset = int(round(self.gate_offset))
            batch_size = enc_out.size(0)

            # Initialize BOS
            bos = get_bos(
                batch_size,
                self.tokens_per_step,
                self.bos_idx,
                device=enc_out.device,
            )
            audio_tokens = bos
            audio_tokens_length = torch.ones(batch_size, device=enc_out.device)
            steps_range = range(self.max_steps)

            # Initialize the gate activation index
            seq_gate_idx = (
                torch.ones(batch_size, device=enc_out.device) * self.max_steps
            )

            # Initialize an indicator that tells whether the gate has activated
            # for a given sample
            seq_gate_act = torch.zeros(batch_size, device=enc_out.device).bool()

            # Show progress if enabled
            if self.show_inference_progress:
                steps_range = tqdm(steps_range, desc="Inference")
            for idx in steps_range:
                # One autoregressive step
                step_out = self.decoder.forward(
                    enc_out=enc_out,
                    src_length=length,
                    tgt=audio_tokens,
                    tgt_length=audio_tokens_length,
                )
                audio_tokens_out = step_out.out.argmax(-1)

                # The model outputs predictions without BOS. Add the BOS back for the
                # following step
                audio_tokens = torch.cat([bos, audio_tokens_out], dim=1)
                # Find the gate activation of the current step
                step_gate_out = step_out.gate_out[:, -1]

                # Compute the gate activation (final sigmoid)
                step_gate_act = step_gate_out.sigmoid() > self.gate_threshold

                # Update the gate activation index as follows
                #
                # - If the gate has already activated in a previous step, leave the index as is
                # - Otherwise:
                #   - If the gate has activated in the current step, update it with the current
                #     step index
                #   - Otherwise, leave it as is
                seq_gate_idx = torch.where(
                    seq_gate_act,
                    seq_gate_idx,
                    torch.where(
                        step_gate_act,
                        torch.tensor(idx, device=step_gate_out.device),
                        seq_gate_idx,
                    ),
                )

                # Update the gate indicator
                seq_gate_act = seq_gate_act | step_gate_act

                # For a given sample, consider it done if the gate has activated at least
                # gate_offset steps ago
                seq_done = seq_gate_act & (idx - seq_gate_idx >= gate_offset)

                # Terminate inference if all samples are done
                done = seq_done.all()
                if done.item():
                    break

            # Length = gate activation index + the offset, not exceeding
            length_abs = (seq_gate_idx + gate_offset).clip(max=self.max_steps)
            max_inferred_len = length_abs.max().int()
            audio_tokens_out = (
                audio_tokens_out[:, :max_inferred_len] - self.audio_token_shift
            )
            # Compute relative lengths
            length = length_abs.float() / audio_tokens_out.size(1)

        return TokotronDecoderInfernceOutput(
            audio_tokens=audio_tokens_out,
            length=length,
            dec_self_attn=step_out.dec_self_attn,
            dec_attn=step_out.dec_attn,
            alignments=step_out.alignments,
            p_eos=step_out.gate_out.sigmoid(),
        )


class TokotronSearchWrapper(nn.Module):
    """A wrapper class to facilitate seach-based inference. It takes care of re-interpreting
    a multi-headed sequence as multiple samples, for compatibility, and for the retention
    of attention tensors

    Arguments
    ---------
    decoder : TokotronTransformerDecoder
        the Tokotron transformer decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.tokens_per_step = decoder.tokens_per_step
        self.decoder = decoder

    def decode(self, memory, enc_states, enc_lens):
        """Wraps the decode operation, will all the necessary
        reshaping

        Arguments
        ---------
        memory : torch.Tensor
            Characters predicted so far
        enc_states : torch.Tensor
            Encoder states
        enc_lens : torch.Tensor
            Encoder state lengths
        """
        batch_size = enc_states.size(0) // self.tokens_per_step
        _, mem_len = memory.shape
        memory = memory.reshape(
            self.tokens_per_step, batch_size, mem_len
        ).permute(1, 2, 0)
        dec_out, dec_self_attn, dec_attn = self.decoder.decode(
            enc_out=enc_states[:batch_size],
            src_length=enc_lens[:batch_size],
            tgt=memory,
        )
        self.dec_self_attn = dec_self_attn
        self.dec_attn = dec_attn
        return dec_out, dec_attn


class TokotronTransformerBeamSearcher(S2STransformerBeamSearcher):
    """A slight modification of S2STransformerBeamSearcher that uses an
    explicit number of tokens instead of trying to infer it from the
    weights of the linear layer. This is needed because Tokotron is
    multi-header and the final output layer outputs multiple output states

    Arguments
    ---------
    num_tokens : int
        The number of audio tokens available
    """

    def __init__(self, num_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens

    def set_n_out(self):
        """Set the number of output tokens."""
        return self.num_tokens


class SearchLinearWrapper(nn.Module):
    """A wrapper for the final linear layer of the Transformer. The goal is to
    make it compatible with the SpeechBrain Beam Search implementation, which is
    single-headed, by expanding multiple heads along the batch dimensions.

    Arguments
    ---------
    lin : torch.Tensor
        A linear layer with an output feature dimensions of
        (tokens_per_step x num_tokens)
    tokens_per_step : int
        the numer of tokens the model outputs for each
        time step
    """

    def __init__(self, lin, tokens_per_step):
        super().__init__()
        self.lin = lin
        self.tokens_per_step = tokens_per_step

    def forward(self, x):
        """Performs a forward pass with all the required reshape operations

        Arguments
        ---------
        x : torch.Tensor
            The decoder output

        Returns
        -------
        result : torch.Tensor
            The layer output, reshaped along the batch dimension
        """
        x = self.lin(x)
        batch_size, max_len, out_dim = x.shape
        num_tokens = x.size(-1) // self.tokens_per_step
        x = (
            # batch x tokens x length
            x.transpose(2, 1)
            # batch x heads x tokens x length
            .view(batch_size, self.tokens_per_step, num_tokens, max_len)
            # heads x batch x tokens x length
            .transpose(0, 1)
            # heads * batch x tokens x length
            .reshape(self.tokens_per_step * batch_size, num_tokens, max_len)
            # heads * batch x length x tokens
            .transpose(1, 2)
        )
        return x


class TokotronSearchInference(nn.Module):
    """A beam search-based inference implementation

    All keyword arguments will be passed on to the underlying
    beam search
    """

    def __init__(self, audio_token_shift=1, **kwargs):
        super().__init__()
        self.search_kwargs = kwargs
        self.audio_token_shift = audio_token_shift
        self.decoder, self.search, self.tokens_per_step = None, None, None

    def bind(self, model=None):
        """Binds this inference implementation to a model

        Arguments
        ---------
        model : TokotronTransformerModel
            The transformer model
        """
        decoder = model.decoder
        self.tokens_per_step = decoder.tokens_per_step
        self.decoder = TokotronSearchWrapper(decoder)
        self.search = TokotronTransformerBeamSearcher(
            modules=[
                self.decoder,
                SearchLinearWrapper(decoder.out_proj, self.tokens_per_step),
            ],
            num_tokens=decoder.num_tokens + self.audio_token_shift,
            **self.search_kwargs,
        )

    def decode(self, enc_out, length):
        """"Decodes the encoder representation using Beam Search

        Arguments
        ---------
        enc_out : torch.Tensor
            Encoder output
        length : torch.Tensor
            Encoder output lengths

        Returns
        -------
        output : TokotronDecoderInfernceOutput
            The inference output
        """
        with torch.no_grad():
            device = enc_out.device
            # The search does not support multiple heads. "Trick" it by expanding encoded
            # representations along the batch dimension so that the beam searcher
            # treats it as if they were separate, independent samples.
            batch_size, max_len, enc_dim = enc_out.shape
            enc_out_search = (
                enc_out.unsqueeze(0)
                .expand(self.tokens_per_step, batch_size, max_len, enc_dim)
                .reshape(self.tokens_per_step * batch_size, max_len, enc_dim)
            )
            length_search = (
                length.unsqueeze(0)
                .expand(self.tokens_per_step, batch_size)
                .reshape(self.tokens_per_step * batch_size)
            )
            hyps, audio_length, scores, log_probs = self.search(
                enc_out_search, length_search
            )
            tokens_batch = PaddedBatch(
                [
                    {"hyps": torch.tensor(item, device=enc_out.device)}
                    for item in hyps
                ]
            ).to(device)

            audio_tokens, length = tokens_batch.hyps
            _, audio_max_len = audio_tokens.shape
            audio_tokens = audio_tokens.reshape(
                self.tokens_per_step, batch_size, audio_max_len
            ).permute(1, 2, 0)
            length = (
                length.reshape(self.tokens_per_step, batch_size).min(dim=0)
            ).values
            audio_tokens = audio_tokens - self.audio_token_shift

            return TokotronDecoderInfernceOutput(
                audio_tokens=audio_tokens,
                length=length,
                dec_self_attn=self.decoder.dec_self_attn,
                dec_attn=self.decoder.dec_attn,
                alignments=get_alignments(self.decoder.dec_attn),
                p_eos=None,
            )


class TokotronForwardInference(nn.Module):
    """A beam search-based inference implementation

    All keyword arguments will be passed on to the underlying
    beam search

    Arguments
    ---------
    scale_factor : float
        The scaling factor for encoder representations
    gate_threshold : float
        The threshold for gate activation
    min_length : int
        The minimum length for generating sequences, in tokens
    """

    def __init__(
        self,
        scale_factor=5.0,
        gate_threshold=0.5,
        min_length=16,
        eos_mode=EosMode.GATE,
        eos_index=0,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.gate_threshold = gate_threshold
        self.min_length = min_length
        self.decoder = None
        self.gate = None
        self.eos_mode = EosMode(eos_mode)
        self.eos_index = eos_index

    def bind(self, model=None):
        """Binds this inference implementation to a model

        Arguments
        ---------
        model : TokotronTransformerModel
            The transformer model
        """
        self.decoder = model.decoder

    def decode(self, enc_out, length):
        """"Decodes the encoder representation using Beam Search

        Arguments
        ---------
        enc_out : torch.Tensor
            Encoder output
        length : torch.Tensor
            Encoder output lengths

        Returns
        -------
        output : TokotronDecoderInfernceOutput
            The inference output
        """
        with torch.no_grad():
            max_len = enc_out.size(1)
            src_key_padding_mask = length_to_mask(
                length * max_len, max_len,
            ).logical_not()
            tgt = scale(enc_out, self.scale_factor)
            dec_out = self.decoder(
                enc_out=enc_out,
                tgt=tgt,
                tgt_length=length,
                src_length=length,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs_src=None,
            )
            if self.eos_mode == EosMode.GATE:
                p_eos, eos = self.get_length_gate(dec_out)
            else:
                p_eos, eos = self.get_length_token(dec_out)

            infer_length_abs = eos.max(dim=1).indices
            infer_length_abs_nonzero = infer_length_abs[infer_length_abs > 0]
            if len(infer_length_abs_nonzero) > 0:
                infer_length_max = infer_length_abs_nonzero.max()
            else:
                infer_length_max = 0
            if infer_length_max == 0:
                infer_length_max = p_eos.size(1)
            infer_length_abs = torch.where(
                infer_length_abs == 0, infer_length_max, infer_length_abs
            )
            infer_length_abs = infer_length_abs.clip(min=self.min_length)
            infer_length = infer_length_abs / infer_length_max

            audio_tokens = dec_out.out[:, :infer_length_max].argmax(-1)
            return TokotronDecoderInfernceOutput(
                audio_tokens=audio_tokens,
                length=infer_length,
                dec_self_attn=dec_out.dec_self_attn,
                dec_attn=dec_out.dec_attn,
                alignments=get_alignments(dec_out.dec_attn),
                p_eos=p_eos,
            )

    def get_length_gate(self, dec_out):
        """Infers lengths using the gate module

        Arguments
        ---------
        dec_out : TokotronDecoderOutput
            The decoder output

        Returns
        -------
        p_eos : torch.Tensor
            EOS probabilities (as estimated by the gate)
        eos : torch.Tensor
            a Boolean tensor where positions indicate whether
            the gate has activated
        """
        p_eos = dec_out.gate_out.sigmoid()
        eos = p_eos > self.gate_threshold
        return p_eos, eos

    def get_length_token(self, dec_out):
        """Infers lengths using an EOS token

        Arguments
        ---------
        dec_out : TokotronDecoderOutput
            The decoder output
        eos : torch.Tensor
            A Boolean tensor indicating whether EOS has been reached
        """
        p_seq = dec_out.out[:, :, 0].softmax(dim=-1)
        p_eos = p_seq[:, :, self.eos_index].softmax(-1)
        eos = p_seq.argmax(dim=-1) == self.eos_index
        return p_eos, eos


class TokotronTransformerModel(nn.Module):
    """An end-to-end Tokotron model receiving characters or phonemes
    as inputs and outputting audio tokens

    Arguments
    ---------
    input_num_tokens : int
        The number of input characters or phonemes available
    audio_num_tokens : int, optional
        The number of audio tokens
    audio_tokens_per_step : int, optional
        The number of output audio tokens per tranformer step.
        When using Vocodec, this corresponds to the number of
        quantizers in the model used
    d_model : int, optional
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    attention_type : str, optional
        The type of attention to be used
    enc_num_layers : int, optional
        The number of encoder layers in1ì the encoder.
    dec_num_layers : int, optional
        The number of decoder layers in the decoder.
    dropout : int, optional
        The dropout value.
    target_dropout : float, optional
        The dropout probability for targets
    activation : torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    max_audio_length: int
        The maximum number of tokens to be output
    infer_max_audio_length: int
        The maximum number of tokens to be output, during inference
    bos_idx : int, optional
        the Beginning-of-Sequence index
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    use_tgt_padding_mask : bool, optional
        Whether to use a target padding mask
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
    show_inference_progress : bool, optional
        Whether to show inference progress in the console
    vocoder : nn.Module
        The vocoder module
    compression_model : nn.Module
        The token compression model to be used
    eos_mode : EosMode | str, optional
        the way the end of sequence is computed
    inference : TokotronInference, optional
        the inference method to be used
    audio_token_shift : int, optional
        The number by which token indices will be shifted (used to introduce
        additional tokens)
    decoder_mode : DecoderMode | str, optional
        The decoding mode (autoregressive or forward)
    scale_factor : float, optional
        forward decoding only - the scaling factor for
        targets in non-autoregressive inference
    """

    def __init__(
        self,
        input_num_tokens,
        audio_num_tokens=1024,
        audio_tokens_per_step=2,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        attention_type="regularMHA",
        enc_num_layers=6,
        dec_num_layers=6,
        dropout=0.2,
        target_dropout=0.2,
        activation=nn.LeakyReLU,
        max_audio_length=1000,
        infer_max_audio_length=None,
        bos_idx=0,
        gate_threshold=0.5,
        gate_offset=0,
        use_tgt_padding_mask=False,
        audio_emb_size=128,
        audio_emb_freeze=False,
        show_inference_progress=True,
        vocoder=None,
        compression_model=None,
        eos_mode=EosMode.GATE,
        inference=None,
        audio_token_shift=0,
        decoder_mode=DecoderMode.AUTOREGRESSIVE,
        scale_factor=5.0,
    ):
        super().__init__()
        self.in_emb = Embedding(
            num_embeddings=input_num_tokens, embedding_dim=d_model,
        )
        self.eos_mode = EosMode(eos_mode)
        self.audio_token_shift = 1 if eos_mode == EosMode.TOKEN else 0
        self.encoder = TransformerEncoder(
            num_layers=enc_num_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            dropout=dropout,
            activation=activation,
            normalize_before=True,
        )
        self.decoder_mode = DecoderMode(decoder_mode)
        audio_emb = None
        if self.decoder_mode == DecoderMode.FORWARD:
            audio_emb = nn.Identity()
            audio_emb_size = d_model
        self.decoder = TokotronTransformerDecoder(
            num_tokens=audio_num_tokens + self.audio_token_shift,
            tokens_per_step=audio_tokens_per_step,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=dec_num_layers,
            activation=activation,
            dropout=dropout,
            target_dropout=target_dropout,
            use_tgt_padding_mask=use_tgt_padding_mask,
            audio_emb=audio_emb,
            audio_emb_size=audio_emb_size,
            audio_emb_freeze=audio_emb_freeze,
            max_decoder_steps=max_audio_length,
            infer_max_decoder_steps=infer_max_audio_length or max_audio_length,
            bos_idx=bos_idx,
            gate_threshold=gate_threshold,
            gate_offset=gate_offset,
            show_inference_progress=show_inference_progress,
            audio_token_shift=audio_token_shift,
            multihead_input=self.decoder_mode == DecoderMode.AUTOREGRESSIVE,
        )
        self.bos_idx = bos_idx
        self.vocoder = vocoder
        self.attention_type = attention_type
        self.gate_offset = gate_offset
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_audio_length
            )
        self.compression_model = compression_model

        if inference is None:
            inference = TokotronTransformerAutoregressiveInference(
                gate_offset=self.gate_offset,
                gate_threshold=gate_threshold,
                tokens_per_step=audio_tokens_per_step,
                bos_idx=bos_idx,
                max_steps=infer_max_audio_length,
                audio_token_shift=self.audio_token_shift,
                show_inference_progress=self.show_inference_progress,
            )
        elif callable(inference) and not isinstance(inference, nn.Module):
            inference = inference()
        self.inference = inference
        self.inference.bind(self)
        self.scale_factor = scale_factor

    def __setattr__(self, name, value):
        """Prevents the vocoder from being saved in state_dict() - it is not typically fine-tuned
        and fine-tuning it would not be trivial

        Arguments
        ---------
        name : str
            The attribute name
        value : any
            The attribute value
        """
        if name in IGNORE_IN_STATE_DICT:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        Arguments
        ---------
        state_dict : dict
            A dict containing parameters and persistent buffers.
        strict : (bool, optional)
            Whether to strictly enforce that the keys
        assign (bool, optional): whether to assign items in the state
            dictionary to their corresponding keys in the module

        Returns
        -------
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
        """
        state_dict = _filter_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except TypeError:
            # NOTE: Older versions of PyTorch don't have the assign parameter
            return super().load_state_dict(state_dict, strict)

    @property
    def gate_offset(self):
        """The number of steps following gate activation to include"""
        return self.decoder.gate_offset

    @gate_offset.setter
    def gate_offset(self, value):
        """The number of steps following gate activation to include"""
        self.decoder.gate_offset = value

    @property
    def show_inference_progress(self):
        """Whether inference progress is displayed"""
        return self.decoder.show_inference_progress

    @show_inference_progress.setter
    def show_inference_progress(self, value):
        """Enables or disables progress display"""
        self.decoder.show_inference_progress = value

    def forward(
        self, input_tokens, input_length, audio_tokens, audio_length,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        audio_tokens : torch.Tensor
            a (Batch x Length) tensor of output audio tokens (e.g. encodec)
        audio_length : torch.Tensor
            a 1-D tensor of relative output lengths"""

        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )

        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        if self.decoder_mode == DecoderMode.AUTOREGRESSIVE:
            tgt = audio_tokens
            tgt_length = audio_length
        else:
            tgt = scale(enc_out, self.scale_factor)
            tgt_length = input_length

        dec_out = self.decoder(
            enc_out=enc_out,
            tgt=tgt,
            tgt_length=tgt_length,
            src_length=input_length,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs_src=pos_embs_encoder,
        )
        return TokotronOutput(
            out=dec_out.out,
            gate_out=dec_out.gate_out,
            p_eos=dec_out.gate_out.sigmoid(),
            enc_self_attn=enc_self_attn,
            dec_self_attn=dec_out.dec_self_attn,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
        )

    def process_inputs(self, input_tokens, input_length):
        """Computes embeddings, the padding mask and encoder
        positional embeddings

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

        Returns
        -------
        src : torch.Tensor
            input embeddings
        src_key_padding_mask : torch.Trnsor
            the key padding mask for inputs
        pos_emb_encoder : torch.Tensor
            encoder positional embeddings
        """
        in_emb = self.in_emb(input_tokens)
        pos_embs_encoder = None
        if self.attention_type == "RelPosMHAXL":
            src = in_emb
            pos_embs_encoder = self.positional_encoding(in_emb)
        else:
            src = in_emb + self.positional_encoding(
                in_emb
            )  # add the encodings here
            pos_embs_encoder = None

        input_max_len = input_tokens.size(1)
        src_key_padding_mask = length_to_mask(
            input_length * input_max_len, input_max_len,
        ).logical_not()
        return src, src_key_padding_mask, pos_embs_encoder

    def infer(self, input_tokens, input_length):
        """Performs end-to-end inference

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
        wav : torch.Tensor
            Synthesized waveforms, if a vocoder is provided
        wav_length : torch.Tensor
            Waveform lengths
        enc_self_attn : torch.Tensor
            Encoder self-attentions
        dec_self_attn : torch.Tensor
            Decoder self-attentions
        dec_attn : torch.Tensor
            Decoder multihead attentions (or equivalent)
        alignments : torch.Tensor
            Aggregated alignments
        p_eos : torch.Tensor
            End-of-sequence probability at each step

        """
        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )
        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        dec_out = self.inference(enc_out, input_length)
        audio_tokens, audio_length = dec_out.audio_tokens, dec_out.length
        wav, wav_length = None, None
        if self.compression_model is not None:
            audio_tokens = self.compression_model.decompress(
                audio_tokens, audio_length
            )
        if self.vocoder is not None:
            vocoder_out = self.vocoder(audio_tokens, dec_out.length)
            if isinstance(vocoder_out, tuple):
                wav, wav_length = vocoder_out
            else:
                wav, wav_length = vocoder_out, dec_out.length
            if wav.dim() == 3:
                wav = wav.squeeze(1)
            clean_padding_(wav, wav_length)
        return TokotronInfernceOutput(
            audio_tokens=audio_tokens,
            length=audio_length,
            wav=wav,
            wav_length=wav_length,
            enc_self_attn=enc_self_attn,
            dec_self_attn=dec_out.dec_self_attn,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
            p_eos=dec_out.p_eos,
        )

    def init_audio_emb(self, emb):
        """Initializes audio embeddings with the specified embedding tensor - useful for re-using the
        embeddings from a pre-trained model

        Arguments
        ---------
        emb : torch.Tensor
            The embedding tensor with which to initialize
        """
        self.decoder.init_audio_emb(emb)


def get_bos(batch_size, tokens_per_step, bos_idx, device="cpu"):
    """Constructs a beginning-of-sequence (BOS) sequence for
    autoregressive inference

    Arguments
    ---------
    batch_size : int
        The size of the batch dimension
    device : str|torch.Device
        The device identifier

    Returns
    -------
    seq: torch.Tensor
        the target sequence"""
    return torch.ones(batch_size, 1, tokens_per_step, device=device) * bos_idx


def get_gate_targets(lengths, out_len):
    """Computes gate tarets and weights for each position

    Arguments
    ---------
    lengths : torch.Tensor
        Relative lengths
    out_len: int
        The maximum output length

    Returns
    -------
    tagrets : torch.Tensor
        Targets for gate outputs - EOS positions are marked as 1,
        non-EOS positions are marked at 0
    weights : torch.Tensor
        Weights by which individual position losses will be multiplied
    """
    pos = torch.arange(out_len, device=lengths.device)[None, :]
    gate_targets = pos >= (lengths * out_len)[:, None]
    gate_weights = torch.where(
        gate_targets, 0.5 / (1.0 - lengths)[:, None], 0.5 / lengths[:, None],
    )
    return gate_targets.float(), gate_weights


def get_alignments(attn):
    """Aggregates alignments from multiple layers and heads

    Arguments
    ---------
    attn: list
        raw attentions returned from a Transformer

    Results
    -------
    alignments: torch.Tensor
        The resulting alignments
    """
    return torch.cat([item.unsqueeze(-1) for item in attn], dim=-1).mean(dim=-1)


TokotronLossDetails = namedtuple(
    "TokotronLossDetails", ["loss", "seq_loss", "gate_loss", "attn_loss"]
)


class TokotronLoss(nn.Module):
    """The loss module for the Tokotron module, combining
    a sequence loss a guided attention loss and a gate loss
    for end-of-sequence prediction

    Arguments
    ---------
    guided_attention_weight : float
        The relative weight of the guided attention loss
    guided_attention_sigma : float
        The sigma hyperparameter for the guided attention loss
        A higher sigma means a lower penalties for attention off
        the diagonal
    gate_weight : float
        The weight of the gate loss
    gate_beta : float
        The beta parameter for the distance difference loss
        used for the EOS gate

        See speechbrain.nnet.losses.distance_diff_loss
        - the beta parameter
    gate_gamma : float
        The gamma parameter for the distance difference loss
        used for the EOS gate

        See speechbrain.nnet.losses.distance_diff_loss
        - the gamma parameter
    gate_max_weight : float
        The maximum distance difference loss weight

        See speechbrain.nnet.losses.distance_diff_loss
        - the max_weight parameter

    silence_padding : float
        The amount of silence padding added to sequences

    seq_cost : float
        The type of sequence loss to be used
    """

    def __init__(
        self,
        guided_attention_weight,
        guided_attention_sigma,
        gate_weight,
        gate_beta,
        gate_gamma,
        gate_max_weight=1.0,
        silence_padding=0,
        seq_cost=None,
        eos_mode=EosMode.GATE,
        audio_token_shift=0.0,
        eos_index=0,
        eos_width=1,
        audio_tokens_per_step=1,
    ):
        super().__init__()
        self.guided_attention_weight = guided_attention_weight
        self.gate_weight = gate_weight
        self.gate_beta = gate_beta
        self.gate_gamma = gate_gamma
        self.gate_max_weight = gate_max_weight
        self.silence_padding = silence_padding
        if seq_cost is None:
            seq_cost = kldiv_loss
        self.seq_cost = seq_cost
        self.attn_cost = GuidedAttentionLoss(sigma=guided_attention_sigma,)
        self.eos_mode = EosMode(eos_mode)
        self.audio_token_shift = audio_token_shift
        self.eos_index = eos_index
        self.eos_width = eos_width
        if self.eos_mode == EosMode.TOKEN:
            audio_eos = (
                torch.ones(eos_width, audio_tokens_per_step).long() * eos_index
            )
            self.register_buffer("audio_eos", audio_eos)

    def forward(
        self,
        predictions,
        audio_tokens,
        audio_length,
        input_tokens,
        input_length,
        reduction="mean",
    ):
        p_seq = predictions.out.log_softmax(dim=-1)
        batch_size, out_len, heads, tok_dim = p_seq.shape
        max_len = out_len - 1
        p_seq_reshaped = (
            p_seq.transpose(1, 2).reshape(batch_size * heads, out_len, tok_dim)
        )[:, :max_len]
        if self.eos_mode == EosMode.TOKEN:
            # NOTE: Shift only the tokens, but not EOS
            padding_lengths = torch.ones(batch_size, device=audio_tokens.device)
            audio_eos = self.audio_eos.unsqueeze(0).expand(
                batch_size, self.eos_width, heads
            )
            audio_tokens, audio_length = concat_padded_features(
                [audio_tokens + self.audio_token_shift, audio_eos],
                [audio_length, padding_lengths],
                dim=1,
            )
            audio_tokens = audio_tokens

        tok_len = audio_tokens.size(1)
        audio_tokens_reshaped = audio_tokens.transpose(1, 2).reshape(
            batch_size * heads, tok_len
        )[:, :max_len]
        lengths_reshaped = (
            audio_length.unsqueeze(-1)
            .expand(batch_size, heads)
            .reshape(batch_size * heads)
        )
        seq_loss = self.seq_cost(
            p_seq_reshaped[:, :tok_len],
            audio_tokens_reshaped,
            length=lengths_reshaped,
            reduction=reduction,
        )
        if reduction == "batch":
            seq_loss = seq_loss.reshape(batch_size, heads).mean(-1)
        lengths_abs = audio_length * out_len

        attn_loss = self.attn_cost(
            predictions.alignments,
            input_lengths=input_length * input_tokens.size(1),
            target_lengths=lengths_abs,
            reduction=reduction,
        )
        if self.eos_mode == EosMode.GATE:
            # NOTE: This adjustment will allow the gate to be "off" by up to silence_padding,
            # resulting in extra silence being output
            gate_loss = distance_diff_loss(
                predictions.p_eos,
                lengths_abs - self.silence_padding,
                beta=self.gate_beta,
                gamma=self.gate_gamma,
                max_weight=self.gate_max_weight,
                two_sided=True,
                reduction=reduction,
            )
        else:
            if reduction == "batch":
                gate_loss = torch.zeros(
                    (batch_size,), device=predictions.out.device
                )
            else:
                gate_loss = torch.tensor(0.0, device=predictions.out.device)
        loss = (
            seq_loss
            + self.guided_attention_weight * attn_loss
            + self.gate_weight * gate_loss
        )
        return TokotronLossDetails(loss, seq_loss, gate_loss, attn_loss)


def _filter_state_dict(state_dict):
    """Removes ignored keys from state_dict.

    Arguments
    ---------
    state_dict : dict
        the raw state_dict

    Returns
    -------
    result : dict
        the filtered state_dict
    """
    return {
        key: value
        for key, value in state_dict.items()
        if not any(
            key.startswith(ignored_key + ".")
            for ignored_key in IGNORE_IN_STATE_DICT
        )
    }


def scale(seq, factor):
    """Scales representations by a factor, in the time dimension only.
    Used in non-autoregressive inference

    Arguments
    ---------
    seq : torch.Tensor
        The sequence to be scaled
    factor : torch.Tensor
        The factor by which teh """
    return F.interpolate(
        seq.unsqueeze(1), scale_factor=(factor, 1), mode="nearest",
    ).squeeze(1)


def distance_diff_loss(
    predictions,
    targets,
    length=None,
    beta=0.25,
    max_weight=100.0,
    gamma=1.0,
    two_sided=False,
    reduction="mean",
):
    """A loss function that can be used in cases where a model outputs
    an arbitrary probability distribution for a discrete variable on
    an interval scale, such as the length of a sequence, and the ground
    truth is the precise values of the variable from a data sample.

    The loss is defined as
    loss_i = p_i * (exp(beta * |i - y|) - 1.) * gamma

    The loss can also be used where outputs aren't probabilities, so long
    as high values close to the ground truth position and low values away
    from it are desired

    Arguments
    ---------
    predictions : torch.Tensor
        a (batch x max_len) tensor in which each element is a probability,
        weight or some other value at that position

    targets : torch.Tensor
        a 1-D tensor in which each elemnent is thr ground truth

    length : torch.Tensor
        lengths (for masking in padded batches)

    beta : float
        a hyperparameter controlling the penalties, an exponent multiplier.
        With a higher beta, penalties will increase faster

    max_weight: torch.Tensor
        the maximum distance weight (for numerical stability in long sequences)

    gamma : float
        a global multiplier - used control the shape of the weighting function

    two_sided : bool
        if set to true, a penalty is added for outputting a low probability
        close to the end

    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size

    Example
    -------
    >>> predictions = torch.tensor(
    ...    [[0.25, 0.5, 0.25, 0.0],
    ...     [0.05, 0.05, 0.9, 0.0],
    ...     [8.0, 0.10, 0.05, 0.05]]
    ... )
    >>> targets = torch.tensor([2., 3., 1.])
    >>> length = torch.tensor([.75, .75, 1.])
    >>> loss = distance_diff_loss(predictions, targets, length)
    >>> loss
    tensor(0.2967)
    """
    return compute_masked_loss(
        partial(
            _distance_diff_loss,
            beta=beta,
            max_weight=max_weight,
            two_sided=two_sided,
            gamma=gamma,
        ),
        predictions=predictions,
        targets=targets,
        length=length,
        reduction=reduction,
        mask_shape="loss",
    )


def distance_diff_loss_ramp(beta, max_weight, gamma):
    """For distance_diff_loss, calculates the number of steps from the ground truth
    at which the weight reaches the maximum


    beta : float
        a hyperparameter controlling the penalties. With a higher beta,
        penalties will increase faster


    max_weight: torch.Tensor
        the maximum distance loss weight

    gamma : float
        a global linear multiplier - used control the shape of the weighting
        function

    """
    return math.log(max_weight / gamma - 1) / beta


def _distance_diff_loss(
    predictions, targets, beta, max_weight, gamma, two_sided=False
):
    """Computes the raw (unreduced) distance difference loss

    Arguments
    ---------
    predictions: torch.Tensor
        a (batch x max_len) tensor in which each element is a probability,
        weight or some other value at that position

    targets: torch.Tensor
        a 1-D tensor in which each elemnent is thr ground truth

    beta: torch.Tensor
        a hyperparameter controlling the penalties. With a higher beta,
        penalties will increase faster

    max_weight: torch.Tensor
        the maximum distance weight (for numerical stability in long sequences)

    gamma : float
        a global multiplier - used control the shape of the weighting function

    two_sided : bool
        if set to true, a penalty is added for outputting a low probability
        close to the end

    """
    batch_size, max_len = predictions.shape
    pos_range = (torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)).to(
        predictions.device
    )
    diff_range = (pos_range - targets.unsqueeze(-1)).abs()
    loss_weights = (((beta * diff_range).exp() - 1.0) * gamma).clamp(
        max=max_weight
    )
    loss = loss_weights * predictions
    if two_sided:
        flip_loss = (max_weight - loss_weights) * (1 - predictions)
        loss = loss + flip_loss
    return loss


# NOTE: GuidedAttentionLoss is included in the SpeechBrain core; however, that version does not
# support the "reduction" argument, which is required by Tokotron, and modifying the core
# is not allowed for the Benchmark
class GuidedAttentionLoss(nn.Module):
    """
    A loss implementation that forces attention matrices to be
    near-diagonal, imposing progressively larger penalties for paying
    attention to regions far away from the diagonal). It is useful
    for sequence-to-sequence models in which the sequence of outputs
    is expected to corrsespond closely to the sequence of inputs,
    such as TTS or G2P

    https://arxiv.org/abs/1710.08969

    The implementation is inspired by the R9Y9 DeepVoice3 model
    https://github.com/r9y9/deepvoice3_pytorch

    It should be roughly equivalent to it; however, it has been
    fully vectorized.

    Arguments
    ---------
    sigma:
        the guided attention weight

    Example
    -------
    NOTE: In a real scenario, the input_lengths and
    target_lengths would come from a data batch,
    whereas alignments would come from a model
    >>> import torch
    >>> from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
    >>> loss = GuidedAttentionLoss(sigma=0.2)
    >>> input_lengths = torch.tensor([2, 3])
    >>> target_lengths = torch.tensor([3, 4])
    >>> alignments = torch.tensor(
    ...     [
    ...         [
    ...             [0.8, 0.2, 0.0],
    ...             [0.4, 0.6, 0.0],
    ...             [0.2, 0.8, 0.0],
    ...             [0.0, 0.0, 0.0],
    ...         ],
    ...         [
    ...             [0.6, 0.2, 0.2],
    ...             [0.1, 0.7, 0.2],
    ...             [0.3, 0.4, 0.3],
    ...             [0.2, 0.3, 0.5],
    ...         ],
    ...     ]
    ... )
    >>> loss(alignments, input_lengths, target_lengths)
    tensor(0.1142)
    """

    def __init__(self, sigma=0.2):
        super().__init__()
        self.sigma = sigma
        self.weight_factor = 2 * (sigma ** 2)

    def forward(
        self,
        attention,
        input_lengths,
        target_lengths,
        max_input_len=None,
        max_target_len=None,
        reduction="mean",
    ):
        """
        Computes the guided attention loss for a single batch

        Arguments
        ---------
        attention: torch.Tensor
            A padded attention/alignments matrix
            (batch, targets, inputs)
        input_lengths: torch.tensor
            A (batch, lengths) tensor of input lengths
        target_lengths: torch.tensor
            A (batch, lengths) tensor of target lengths
        max_input_len: int
            The maximum input length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        max_target_len: int
            The maximum target length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        reduction : str
            The loss reduction.
            Supported: "batch" or "mean"


        Returns
        -------
        loss: torch.Tensor
            A single-element tensor with the loss value
        """
        soft_mask = self.guided_attentions(
            input_lengths, target_lengths, max_input_len, max_target_len
        )
        loss = attention * soft_mask.transpose(-1, -2)
        if reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.mean([-1, -2])
        return loss

    def guided_attentions(
        self,
        input_lengths,
        target_lengths,
        max_input_len=None,
        max_target_len=None,
    ):
        """
        Computes guided attention matrices

        Arguments
        ---------
        input_lengths: torch.Tensor
            A tensor of input lengths
        target_lengths: torch.Tensor
            A tensor of target lengths
        max_input_len: int
            The maximum input length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        max_target_len: int
            The maximum target length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism

        Returns
        -------
        soft_mask: torch.Tensor
            The guided attention tensor of shape (batch, max_input_len, max_target_len)
        """
        input_lengths_broad = input_lengths.view(-1, 1, 1)
        target_lengths_broad = target_lengths.view(-1, 1, 1)
        if max_input_len is None:
            max_input_len = input_lengths.max()
        if max_target_len is None:
            max_target_len = target_lengths.max()
        input_mesh, target_mesh = torch.meshgrid(
            torch.arange(max_input_len).to(input_lengths.device),
            torch.arange(max_target_len).to(target_lengths.device),
        )
        input_mesh, target_mesh = (
            input_mesh.unsqueeze(0),
            target_mesh.unsqueeze(0),
        )
        input_lengths_broad = input_lengths.view(-1, 1, 1)
        target_lengths_broad = target_lengths.view(-1, 1, 1)
        soft_mask = 1.0 - torch.exp(
            -(
                (
                    input_mesh / input_lengths_broad
                    - target_mesh / target_lengths_broad
                )
                ** 2
            )
            / self.weight_factor
        )
        outside = (input_mesh >= input_lengths_broad) | (
            target_mesh >= target_lengths_broad
        )
        soft_mask[outside] = 0.0
        return soft_mask
