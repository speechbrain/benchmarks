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
    PositionalEncoding as TransformerPositionalEncoding,
    get_lookahead_mask,
)
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.losses import kldiv_loss, mse_loss, compute_masked_loss, bce_loss
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.data_utils import concat_padded_features
from speechbrain.nnet.schedulers import NoamScheduler
from model.sq_codec import decimal_to_ternary_matrix

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
    ["audio", "length", "dec_self_attn", "dec_attn", "alignments", "p_eos"],
)

TokotronInfernceOutput = namedtuple(
    "TokotronInferenceOutput",
    [
        "audio",
        "length",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
        "p_eos",
    ],
)


class EosMode(Enum):
    GATE = "gate"
    TOKEN = "token"


class DecoderMode(Enum):
    AUTOREGRESSIVE = "autoregressive"
    FORWARD = "forward"


class RepresentationMode(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


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
    audio_emb_size : int
        The size of the audio embeddings (if learned)
    activation : torch.nn.Module, optional
        The activation function to be used
    use_tgt_padding_mask : bool, optional
        whether to use a target padding mask
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
    max_decoder_steps : int, optional
        The maximum number of decoder steps used during training
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    audio_dim : int, optional
        The continuous audio input dimension
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
        bos_width=1,
        gate_threshold=0.5,
        gate_offset=0,
        show_inference_progress=True,
        audio_token_shift=0,
        multihead_input=True,
        multihead_output=True,
        representation_mode=RepresentationMode.DISCRETE,
        audio_dim=1024,
        out_proj=None,
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
        self.representation_mode = RepresentationMode(representation_mode)
        self.out_dim = (
            num_tokens + audio_token_shift
            if self.representation_mode == RepresentationMode.DISCRETE
            else audio_dim
        )
        if out_proj is None:
            out_proj = Linear(
                input_size=d_model, n_neurons=self.out_dim * tokens_per_step,
            )
        self.out_proj = out_proj
        self.gate = Linear(input_size=d_model, n_neurons=1)
        if audio_emb is None:
            if self.representation_mode == RepresentationMode.DISCRETE:
                audio_emb = MultiEmbedding(
                    num_embeddings=num_tokens + audio_token_shift,
                    embedding_dim=audio_emb_size,
                    num_heads=tokens_per_step,
                    normalized=True,
                    d_model=d_model,
                )
            else:
                audio_emb = Linear(
                    input_size=audio_dim, n_neurons=audio_emb_size,
                )

        self.positional_encoding = PositionalEncoding(
            d_model, max_decoder_steps
        )
        if target_dropout is None:
            target_dropout = dropout
        self.target_dropout = target_dropout
        self.audio_emb = audio_emb
        self.max_decoder_steps = max_decoder_steps
        self.attention_type = attention_type
        self.use_tgt_padding_mask = use_tgt_padding_mask
        self.audio_emb_freeze = audio_emb_freeze
        self.bos_width = bos_width
        self.gate_threshold = gate_threshold
        self.gate_offset = gate_offset
        self.show_inference_progress = show_inference_progress
        if self.audio_emb_freeze:
            for parameter in self.audio_emb.parameters():
                parameter.requires_grad_(False)
        self.audio_token_shift = audio_token_shift
        self.multihead_input = multihead_input
        self.d_model = d_model
        self.d_model_sqrt = math.sqrt(d_model)
        self.multihead_output = multihead_output

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
        """Performs a decode step

        Arguments
        ---------
        enc_out : torch.Tensor
            The raw encoder outputs
        tgt : torch.Tensor, optional
            Targets (i.e. the parts of the audio sequence that have
            already been decoded)
        src_length : torch.Tensor, optional
            Relative length of input sequences
        src_key_padding_mask : torch.Tensor, optional
            Key padding mask for the tensor (if pre-computed)
        tgt_length : torch.Tensor, optional
            The target relative length
        tgt_key_padding_mask : torch.Tensor, optional
            The target key padding mask (if pre-computed)
        pos_emb_src : torch.Tensor, optional
            The target positional embeddings

        Returns
        -------
        dec_out : torch.Tensor
            Decoder outputs
        dec_self_attn : list
            Decorder self-attentions (list of tensors)
        dec_attn : list
            Decoder attention (list of tensors)
        """
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
        # NOTE: Normalization for continuous representations, similar
        # to NormalizedEmbedding
        if self.representation_mode == RepresentationMode.CONTINUOUS:
            tgt = tgt * self.d_model_sqrt
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

        Returns
        -------
        result : TokotronDecoderOutput
            The model output

            out : torch.Tensor
                The outputs - token probabilities
                Batch x Length x Head x Token
            gate_out : torch.Tensor
                The gate activation tesnsor
                Batch x Length
            dec_self_attn : list
                Decoder self-attentions
            dec_attn : list
                Decoder multi-head attentions
            alignments : torch.Tensor
                Decoder multi-head attentions, concatenated
                as a single tensor
            context : dict
                An empty dictionary (not used in this decoder)
        """
        if self.representation_mode == RepresentationMode.DISCRETE:
            tgt_shift = torch.zeros((1, tgt.size(1), 1), device=tgt.device)
            tgt_shift[:, self.bos_width :, :] += self.audio_token_shift
            tgt = tgt + tgt_shift
        dec_out, dec_self_attn, dec_attn = self.decode(
            enc_out,
            tgt,
            src_length,
            src_key_padding_mask,
            tgt_length,
            tgt_key_padding_mask,
            pos_embs_src,
        )
        lin_out = self.out_proj(dec_out)
        if self.multihead_output:
            batch_size, audio_max_len, num_tokens = lin_out.shape
            lin_out = lin_out.reshape(
                batch_size,
                audio_max_len,
                self.tokens_per_step,
                num_tokens // self.tokens_per_step,
            )
        gate_out = self.gate(dec_out).squeeze(-1)
        return TokotronDecoderOutput(
            lin_out,
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


class TernaryPredictionHead(nn.Module):
    """An alternative prediction head that predicts a fixed number of ternary digits
    for each position (as used in SQ-Codec)
    
    Arguments
    ---------
    d_model : int
        The model dimension
    num_positions : int
        the number of positions
    """
    def __init__(self, d_model, num_positions):
        super().__init__()
        self.num_positions = num_positions
        self.d_model = d_model
        self.num_positions = num_positions
        self.lin_p = Linear(
            input_size=d_model,
            n_neurons=num_positions * 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Computes the forward pass
        
        Arguments
        ---------
        x : torch.Tensor
            The decoder output (Batch x Length x d_model)

        Returns
        -------
        p : torch.Tensor
            A tensor of shape (Batch x Length x num_positions x 2) where
            p[:, :, :, 0] -> the probability of the ternary digit being at least 0
            p[:, :, :, 0] -> the probability of the ternary digit being at least 1
        """
        batch_size, max_len, _ = x.shape
        p = self.sigmoid(self.lin_p(x))
        p = p.reshape(batch_size, max_len, self.num_positions, 2)
        return p


class TernaryInput(nn.Module):
    def __init__(self, emb_size, num_positions):
        super().__init__()
        self.num_positions = num_positions
        self.in_proj = Linear(
            input_size=num_positions * 3,
            n_neurons=emb_size,
        )

    def forward(self, x):
        batch_size, max_len = x.shape[:2]
        x_onehot = torch.nn.functional.one_hot(
            (x + 1).long(),
            3
        ).reshape(batch_size, max_len, self.num_positions * 3)
        in_proj = self.in_proj(x_onehot.float())
        return in_proj


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
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    audio_dim : int, optional
        The continuous audio input dimension
    show_inference_progress : bool, optional
        Whether to show inference progress in the console
    """

    def __init__(
        self,
        gate_offset,
        gate_threshold,
        tokens_per_step,
        bos_idx,
        max_steps,
        audio_token_shift,
        representation_mode=RepresentationMode.DISCRETE,
        audio_dim=1024,
        show_inference_progress=True,
        transform_audio=None,
        feed_audio=None
    ):
        super().__init__()
        self.decoder = None
        self.gate_offset = gate_offset
        self.gate_threshold = gate_threshold
        self.tokens_per_step = tokens_per_step
        self.bos_idx = bos_idx
        self.max_steps = max_steps
        self.audio_token_shift = audio_token_shift
        self.representation_mode = RepresentationMode(representation_mode)
        self.audio_dim = audio_dim
        self.show_inference_progress = show_inference_progress
        if transform_audio is None:
            transform_audio = nn.Identity()
        self.transform_audio = transform_audio
        self.feed_audio = feed_audio

    def bind(self, model):
        """Binds this inference implementation to a model

        Arguments
        ---------
        model : TokotronTransformerModel
            The transformer model
        """
        self.decoder = model.decoder

    def forward(self, enc_out, length, emb=None):
        """Performs autoregressive inference

        Arguments
        ---------
        decoder : callable
            The decoder module

        enc_out : torch.Tensor
            Raw encoder outputs

        length : torch.Tensor
            Relative lengths

        emb : dict, optional
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)

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
                audio_dim=self.audio_dim,
                representation_mode=self.representation_mode,
                device=enc_out.device,
            )
            audio = bos
            audio_length = torch.ones(batch_size, device=enc_out.device)
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
                audio = self.transform_audio(audio)
                step_out = self.decoder.forward(
                    enc_out=enc_out,
                    src_length=length,
                    tgt=audio,
                    tgt_length=audio_length,
                )
                audio_out = step_out.out

                if self.feed_audio:
                    audio_out = self.feed_audio(audio_out)
                elif self.representation_mode == RepresentationMode.DISCRETE:
                    audio_out = audio_out.argmax(-1)

                # The model outputs predictions without BOS. Add the BOS back for the
                # following step
                audio = torch.cat([bos, audio_out], dim=1)

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
            audio_out = audio_out[:, :max_inferred_len] - self.audio_token_shift
            # Compute relative lengths
            length = length_abs.float() / audio_out.size(1)

        if self.representation_mode == RepresentationMode.CONTINUOUS:
            audio_out = bipolar_compression_inv(audio_out)

        return TokotronDecoderInfernceOutput(
            audio=audio_out,
            length=length,
            dec_self_attn=step_out.dec_self_attn,
            dec_attn=step_out.dec_attn,
            alignments=step_out.alignments,
            p_eos=step_out.gate_out.sigmoid(),
        )


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
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    audio_dim : int, optional
        The continuous audio inout dimension
    emb : dict, optional
        Available embeddings

        Example:
        {
            "spk": {
                "kind": "pretrained"
                "dim" : 512
            },
            "lang": {
                "kind": "trained",
                "count" : 2
            }
        }
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
        eos_mode=EosMode.GATE,
        inference=None,
        audio_token_shift=0,
        scale_factor=5.0,
        representation_mode=RepresentationMode.DISCRETE,
        audio_dim=1024,
        emb=None,
        audio_emb=None,
        out_proj=None,
        multihead_input=True
    ):
        super().__init__()
        self.in_emb = Embedding(
            num_embeddings=input_num_tokens, embedding_dim=d_model,
        )
        self.eos_mode = EosMode(eos_mode)
        self.d_model = d_model
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
            gate_threshold=gate_threshold,
            gate_offset=gate_offset,
            audio_token_shift=audio_token_shift,
            multihead_input=multihead_input,
            multihead_output=out_proj is None,
            representation_mode=representation_mode,
            audio_dim=audio_dim,
            out_proj=out_proj,
        )
        self.bos_idx = bos_idx
        self.attention_type = attention_type
        self.gate_offset = gate_offset
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_audio_length
            )

        if inference is None:
            inference = TokotronTransformerAutoregressiveInference(
                gate_offset=self.gate_offset,
                gate_threshold=gate_threshold,
                tokens_per_step=audio_tokens_per_step,
                bos_idx=bos_idx,
                max_steps=infer_max_audio_length,
                audio_token_shift=self.audio_token_shift,
                representation_mode=representation_mode,
                audio_dim=audio_dim,
                show_inference_progress=show_inference_progress,
            )
        elif callable(inference) and not isinstance(inference, nn.Module):
            inference = inference()
        self.inference = inference
        self.inference.bind(self)
        self.scale_factor = scale_factor
        self.representation_mode = RepresentationMode(representation_mode)
        self.audio_dim = audio_dim
        if emb is not None:
            self.emb_proj = self._build_emb_proj(emb)

    def _build_emb_proj(self, emb):
        """Builds the embedding projection

        Arguments
        ---------
        emb : dict
            Embedding configuration

        Returns
        -------
        emb_proj : torch.nn.ModuleDict
            embedding projections for each embedding"""
        emb_proj = {}
        for key, emb_config in emb.items():
            kind = emb_config.get("kind", "learned")
            if kind == "pretrained":
                emb_mod = Linear(
                    input_size=emb_config.get("dim", self.d_model)
                    + self.d_model,
                    n_neurons=self.d_model,
                )
            elif kind == "learned":
                emb_count = emb_config["count"]
                emb_dim = emb_config.get("dim", self.d_model)
                emb_mod = Embedding(
                    num_embeddings=emb_count, embedding_dim=emb_dim
                )
            else:
                raise ValueError(f"Invallid embedding kind: {kind}")
            emb_proj[key] = emb_mod
        return nn.ModuleDict(emb_proj)

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
            return super().load_state_dict(state_dict, False, assign)
        except TypeError:
            # NOTE: Older versions of PyTorch don't have the assign parameter
            return super().load_state_dict(state_dict, False)

    @property
    def gate_offset(self):
        """The number of steps following gate activation to include"""
        return self.decoder.gate_offset

    @gate_offset.setter
    def gate_offset(self, value):
        """The number of steps following gate activation to include"""
        self.decoder.gate_offset = value

    def forward(
        self, input_tokens, input_length, audio, audio_length, emb=None
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
            a 1-D tensor of relative output lengths
        emb : dict
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)

        Returns
        -------
        result : TokotronOutput
            Forward step outputs
            out : torch.Tensor
                The outputs - token probabilities
                Batch x Length x Head x Token
            gate_out : torch.Tensor
                The gate activation tesnsor
                Batch x Length
            dec_self_attn : list
                Encoder self-attentions
            dec_self_attn : list
                Decoder self-attentions
            dec_attn : list
                Decoder multi-head attentions
            alignments : torch.Tensor
                Decoder multi-head attentions, concatenated
                as a single tensor
        """

        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )
        if self.representation_mode == RepresentationMode.CONTINUOUS:
            audio = bipolar_compression(audio)

        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        enc_out = self.add_emb(enc_out, emb)
        dec_out = self.decoder(
            enc_out=enc_out,
            tgt=audio,
            tgt_length=audio_length,
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

    def add_emb(self, src, emb):
        """Adds embedding projections to the source tensor

        Arguments
        ---------
        src : torch.Tensor
            The source tensor
        emb : dict
            The embedding dictionary

        Arguments
        ---------
        result : torch.Tensor
            The resulting tensor, with embeddings incorporated
        """
        result = src
        if emb is not None:
            for key, emb_t in emb.items():
                batch_size, seq_len, feat_size = src.shape
                emb_size = emb_t.size(-1)
                emb_t_norm = nn.functional.layer_norm(emb_t, emb_t.shape)
                emb_exp = emb_t_norm.unsqueeze(1).expand(
                    batch_size, seq_len, emb_size
                )
                src_norm = nn.functional.layer_norm(src, src.shape)
                src_with_emb = torch.cat([src_norm, emb_exp], dim=-1)
                result = self.emb_proj[key](src_with_emb)
                src = result
        return result

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

    def infer(self, input_tokens, input_length, emb=None):
        """Performs end-to-end inference

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        emb : dict, optional
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)


        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
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
        enc_out = self.add_emb(enc_out, emb)
        dec_out = self.inference(enc_out, input_length)
        audio, audio_length = dec_out.audio, dec_out.length
        return TokotronInfernceOutput(
            audio=audio,
            length=audio_length,
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


def get_bos(
    batch_size,
    tokens_per_step,
    bos,
    audio_dim=None,
    representation_mode=RepresentationMode.DISCRETE,
    device="cpu",
):
    """Constructs a beginning-of-sequence (BOS) sequence for
    autoregressive inference

    Arguments
    ---------
    batch_size : int
        The size of the batch dimension
    tokens_per_step : int
        The number of tokens per step
    bos : int | float
        The BOS token index for discrete representations or the
        value to be used for continuous representations
    audio_dim : int
        The dimension of audio representations (used if representation_mode is set to
        CONTINUOUS)
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    device : str|torch.Device
        The device identifier

    Returns
    -------
    seq: torch.Tensor
        the sequence consisting only of BOS"""
    if representation_mode == RepresentationMode.DISCRETE:
        seq = torch.ones(batch_size, 1, tokens_per_step, device=device) * bos
    else:
        seq = (
            torch.ones(batch_size, 1, tokens_per_step, audio_dim, device=device)
            * bos
        )
    return seq


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

    audio_tokens_per_step : int
        The number of audio tokens per step

    representation_mode : RepresentationMode
        the type of representations being used (discrete or continuous)

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
        representation_mode=RepresentationMode.DISCRETE,
        audio_clip_min=-10.0,
        audio_clip_max=10.0,
        multihead_output=True,
    ):
        super().__init__()
        self.guided_attention_weight = guided_attention_weight
        self.gate_weight = gate_weight
        self.gate_beta = gate_beta
        self.gate_gamma = gate_gamma
        self.gate_max_weight = gate_max_weight
        self.silence_padding = silence_padding
        self.representation_mode = RepresentationMode(representation_mode)
        if seq_cost is None:
            seq_cost = (
                kldiv_loss
                if self.representation_mode == RepresentationMode.DISCRETE
                else mse_loss
            )
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
        self.audio_clip_min = audio_clip_min
        self.audio_clip_max = audio_clip_max
        self.multihead_output = multihead_output

    def forward(
        self,
        predictions,
        audio,
        audio_length,
        input_tokens,
        input_length,
        reduction="mean",
    ):
        """Computes the loss, with details

        Arguments
        ---------
        predictions : TokotronOutput
            the raw predictions, from the model
        audio : torch.Tensor
            target audio tokens
        audio_length : torch.Tensor
            relative lengths of target audio, for masking
        input_tokens : torch.Tensor
            input tokens (text of phonemes)
        input_length : torch.Tensor
            relative lengths of input tokens, for masking
        reduction : str
            loss reduction (see speechbrain.nnet.losses)
        """
        out = predictions.out
        if self.representation_mode == RepresentationMode.DISCRETE:
            out = out.log_softmax(dim=-1)
        batch_size, out_len, heads, tok_dim = out.shape
        max_len = out_len - 1
        if self.multihead_output:
            out_reshaped = (
                out.transpose(1, 2).reshape(batch_size * heads, out_len, tok_dim)
            )[:, :max_len]
        else:
            out_reshaped = out
        if self.eos_mode == EosMode.TOKEN:
            # NOTE: Shift only the tokens, but not EOS
            padding_lengths = torch.ones(batch_size, device=audio.device)
            audio_eos = self.audio_eos.unsqueeze(0).expand(
                batch_size, self.eos_width, heads
            )
            features = [audio + self.audio_token_shift, audio_eos]
            features = [item.float() for item in features]
            audio, audio_length = concat_padded_features(
                features, [audio_length, padding_lengths], dim=1,
            )

        tok_len = audio.size(1)
        if not self.multihead_output:
            audio_reshaped = audio
            lengths_reshaped = audio_length
        elif self.representation_mode == RepresentationMode.DISCRETE:
            audio_reshaped = audio.transpose(1, 2).reshape(
                batch_size * heads, max_len
            )
        else:
            audio_dim = audio.size(-1)
            audio_reshaped = audio.transpose(1, 2).reshape(
                batch_size * heads, max_len, audio_dim
            )
            audio_reshaped = bipolar_compression(audio_reshaped)
            if (
                self.audio_clip_min is not None
                or self.audio_clip_max is not None
            ):
                audio_reshaped = audio_reshaped.clip(
                    min=self.audio_clip_min, max=self.audio_clip_max,
                )

        audio_reshaped = audio_reshaped[:, :max_len]
        if self.multihead_output:        
            lengths_reshaped = (
                audio_length.unsqueeze(-1)
                .expand(batch_size, heads)
                .reshape(batch_size * heads)
            )
        else:
            lengths_reshaped = audio_length            
        seq_loss = self.seq_cost(
            out_reshaped[:, :tok_len],
            audio_reshaped,
            length=lengths_reshaped,
            reduction=reduction,
        )
        if reduction == "batch" and self.multihead_output:
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
        if not key.endswith(".pe")
    }


def scale(seq, factor):
    """Scales representations by a factor, in the time dimension only.
    Used in non-autoregressive inference

    Arguments
    ---------
    seq : torch.Tensor
        The sequence to be scaled
    factor : torch.Tensor
        The factor by which the inputs will be scaled

    Returns
    -------
    result : torch.Tensor
        The input, scaled by the specified factor
    """
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


def bipolar_compression(x):
    """The bipolar compression function
    f(x) = sign(x) ln(|x| + 1)
    """
    return x.sign() * (x.abs() + 1).log()


def bipolar_compression_inv(x):
    """The inverse of bipolar_compression"""
    return torch.where(x >= 0, x.exp() - 1.0, 1.0 - (-x).exp())


class TargetedNoamScheduler(NoamScheduler):
    """A customization of NoamScheduler that does not assume all parameter groups have the same
    learning rate

    Arguments
    ---------
    lr_initial : list
        Initial learning rate (i.e. the lr used at epoch 0), for each parameter group
    n_warmup_steps : int
        number of warm-up steps
    model_size : int
        size of transformer embed_dim. It is used to scale the maximum learning rate value reached
        by the scheduler. It is divided by model_size ** (0.5).
        If not specified the maximum learning rate value is instead multiplied by warmup_steps ** (0.5).
    """

    def __init__(
        self, lr_initial, n_warmup_steps, model_size=None, param_group=None
    ):
        super().__init__(
            lr_initial=lr_initial,
            n_warmup_steps=n_warmup_steps,
            model_size=model_size,
        )
        self.param_group = param_group

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        # Changing the learning rate within the optimizer
        for param_group, lr_initial in zip(opt.param_groups, self.lr_initial):
            lr = lr_initial * self._get_lr_scale()
            param_group["lr"] = lr

        self.current_lr = current_lr
        lr = opt.param_groups[0]["lr"]
        return current_lr, lr


class PositionalEncoding(TransformerPositionalEncoding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        pass


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


def get_silence_token(
    model,
    sample_length=100000,
    unsqueeze=False,
    device=None,
    num_codebooks=None,

):
    """Attempts to find out the silence tokens for a given model,
    if applicable

    Arguments
    ---------
    model : nn.Module
        A discrete token model, taking (wav, lengths) as arguments
    sample_length : int
        The length of the sample
    unsqueeze: bool
        Whether to add an extra dimension to the audio (needed for DAC)
    device : str | torch.Device
        The device to use
    num_codebooks : int | list
        The number of codebooks or the codebooks to use

    Returns
    -------
    silence_tokens : torch.Tensor
        The token(s) corresponding to silence

    silece_emb : torch.Tensor
        The embedding(s) corresponding to silence

    """
    if device is None:
        device = next(model.parameters()).device

    audio = torch.zeros(1, sample_length, device=device)
    if unsqueeze:
        audio = audio.unsqueeze(1)
    length = torch.ones(1, device=device)
    model_training = model.training
    model.eval()
    tokens = model.sig_to_tokens(audio, length, num_codebooks=num_codebooks)
    if model_training:
        model.train()
    tokens = tokens.squeeze(0)
    if unsqueeze:
        tokens = tokens.squeeze(0)
    silence_tokens = tokens.mode(0).values
    return silence_tokens


def get_silence_repr(model, sample_length=100000, device=None):
    """Gets continuous silence

    Arguments
    ---------
    model : nn.Module
        A discrete token model, taking (wav, lengths) as arguments
    sample_length : int
        The length of the sample
    device : str | torch.Device
        The device to use

    Returns
    -------
    silence : torch.Tensor
        A silecnce tensor
    """
    audio = torch.zeros(1, sample_length, device=device)
    length = torch.ones(1, device=device)
    audio_repr = model(audio, length)
    silence = audio_repr.mean(dim=1)[0]
    return silence


def feature_pad_to(tensor, length, padding=None):
    """Pads feature dimensions to the specified length with the specified padding,
    assuming a (Batch x Length x Features..) tensor

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be padded

    length : int
        The length to which the tensor will be padded

    padding : torch.Tensor, optional
        The padding tensor - if omitted, zero padding
        will be used

    Returns
    -------
    result : torch.Tensor
        The padded tensor
    """
    if padding is None:
        padding = torch.zeros(tensor.shape[1:])
    padding = padding[None, ...].expand(
        (length - tensor.size(0),) + tensor.shape[1:]
    )
    return torch.cat([tensor, padding], dim=0)


def batch_feature_pad(tensors, padding=None):
    """Similar to batch_pad_right but pads with the specified padding, whcih
    can be a vector or a tensor

    Arguments
    ---------
    tensors : list
        The list of tensors to be padded
    padding : torch.Tensor
        The padding tensor

    Returns
    -------
    result : torch.Tensor
        the padded tensor
    """
    lengths_abs = torch.tensor(
        [len(item) for item in tensors], device=tensors[0].device
    )
    max_length = lengths_abs.max()
    data = torch.stack(
        [feature_pad_to(item, max_length, padding) for item in tensors]
    )
    lengths = lengths_abs / max_length
    return data, lengths


def token_collate_fn(examples, silence_token, token_keys):
    """A customized collation function for audio tokens where
    the specified silence token will be used as padding - instead of
    zeros

    Arguments
    ---------
    examples : list
        A list of examples

    silence_token : torch.Tensor
        The token(s) representing silence

    token_keys : list
        The list of keys to which special padding will be applied

    Returns
    -------
    result : speechbrain.dataio.batch.PaddedBatch
        A padded batch
    """
    token_tensor_ids = {id(examples[0][key]) for key in token_keys}
    return PaddedBatch(
        examples,
        padding_func=_silence_padding,
        padding_kwargs={
            "silence_token": silence_token,
            "token_tensor_ids": token_tensor_ids,
        },
    )


def _silence_padding(values, silence_token, token_tensor_ids):
    return (
        batch_feature_pad(values, silence_token)
        if id(values[0]) in token_tensor_ids
        else batch_pad_right(values)
    )


def use_silence_padding(dataloader_opts, silence_token, token_keys):
    """Overrides the collation function to add silence padding to
    audio token features

    Arguments
    ---------
    dataloder_opts : dict
        Dataloader options
    silence_token : torch.Tensor
        The tensor to be used as silence padding
    token_keys : torch.Tensor
        The keys to apply silence padding to

    Returns
    -------
    dataloader_opts : dict
        Updated data loader options
    """
    return {
        **dataloader_opts,
        "collate_fn": partial(
            token_collate_fn, silence_token=silence_token, token_keys=token_keys
        ),
    }


def ternary_matrix_to_decimal(matrix):
    """
    Convert a B*D*N ternary matrix to a 2D array of decimal numbers for each batch.

    Arguments
    ---------
    matrix : numpy.ndarray
        A 3D numpy array of shape (B, D, N), where B is the batch size, D is the number
        of ternary digits, and N is the number of ternary numbers in each batch.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (B, N), where each value represents the decimal
        equivalent of the corresponding ternary number in the input matrix.
    """
    (
        B,
        D,
        N,
    ) = (
        matrix.shape
    )  # B is the batch size, D is the number of digits, N is the number of ternary numbers
    powers_of_three = 3 **torch.arange(D)  # [3^0, 3^1, ..., 3^(D-1)]

    # Reshape powers_of_three for broadcasting: [D] -> [1, D, 1]
    powers_of_three = powers_of_three[:, None]  # Shape [D, 1]

    # Compute dot product using broadcasting: matrix * powers_of_three along D axis
    decimals = torch.sum(matrix * powers_of_three, axis=1)  # Sum along the D axis

    return decimals


def logits_to_ternary(logits):
    """Converts a tensor with two logits to a ternary matrix

    Arguments
    ---------
    logits : torch.Tensor
        The logits (Batch x Length x num_positions x 2)

    Returns
    -------
    result : torch.Tensor
        The corresponding ternary matrix
    """
    gte0 = logits[..., 0] >= 0.5
    gte1 = logits[..., 1] >= 0.5
    val_minus_1 = torch.tensor(-1, device=logits.device)
    val_zero = torch.tensor(0, device=logits.device)
    val_plus_1 = torch.tensor(1, device=logits.device)
    return torch.where(
        gte0,
        torch.where(
            gte1,
            val_plus_1,
            val_zero
        ),
        val_minus_1
    )

def ternary_matrix_to_decimal(matrix):
    """
    Convert a B*D*N ternary matrix to a 2D array of decimal numbers for each batch.

    Arguments
    ---------
    matrix : numpy.ndarray
        A 3D numpy array of shape (B, D, N), where B is the batch size, D is the number
        of ternary digits, and N is the number of ternary numbers in each batch.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (B, N), where each value represents the decimal
        equivalent of the corresponding ternary number in the input matrix.
    """
    (
        B,
        D,
        N,
    ) = (
        matrix.shape
    )  # B is the batch size, D is the number of digits, N is the number of ternary numbers
    powers_of_three = 3 ** torch.arange(D, device=matrix.device)  # [3^0, 3^1, ..., 3^(D-1)]

    # Reshape powers_of_three for broadcasting: [D] -> [1, D, 1]
    powers_of_three = powers_of_three[:, None]  # Shape [D, 1]

    # Compute dot product using broadcasting: matrix * powers_of_three along D axis
    decimals = torch.sum(matrix * powers_of_three, axis=1)  # Sum along the D axis

    return decimals


def ternary_to_decimal(ternary, n_codebook=4):
    """Converts ternary digits to their decimal equivalent
    
    Arguments
    ---------
    ternary : torch.Tensor
        (Batch x Length x num_positions) - ternary digits
    n_codebooks : torch.Tensor
        The number of coedbooks"""
    chunks = ternary.chunk(n_codebook, dim=1)
    codec_ls = []
    # TODO: Vectorize
    for i, chunk in enumerate(chunks):
        chunk = chunk + 1
        tmp_codec = ternary_matrix_to_decimal(chunk)
        codec_ls.append(tmp_codec)
    codec_ls = torch.stack(codec_ls)
    return codec_ls.permute(1, 2, 0)


def ternary_logits_to_tokens(logits):
    """Converts ternary logits to tokens (as used for SQ-Codec)

    Arguments
    ---------
    logits : torch.Tensor
        The logits

    Returns
    -------
    tokens : torch.Tensor
        Token IDs
    """
    ternary_matrix = logits_to_ternary(logits)
    tokens = ternary_to_decimal(ternary_matrix.transpose(-1, -2))
    return tokens


def tokens_to_ternary(tokens):
    """Converts a sequence of tokens to a ternary matrix
    
    Arguments
    ---------
    tokens : torch.Tensor
        A (Batch x Length x Codebooks) tensor of tokens
    
    Returns
    -------
    result : t""" 
    batch_size = tokens.size(0)
    n_codebook = tokens.size(2)
    tokens = tokens.view(batch_size, -1, n_codebook).permute(2, 0, 1).clone()
    ternary_matrix = torch.cat([
        decimal_to_ternary_matrix(item, D=9) - 1
        for item in tokens
    ], dim=1)
    return ternary_matrix.transpose(1, 2)


def ternary_loss(predictions, targets, length=None, reduction="mean"):
    tgt_gte0 = targets >= 0.
    tgt_gte1 = targets >= 1.
    loss_gte0 = bce_loss(
        predictions[:, :, :, 0],
        tgt_gte0,
        length=length,
        reduction=reduction,
    )
    loss_gte1 = bce_loss(
        predictions[:, :, :, 0],
        tgt_gte1,
        length=length,
        reduction=reduction,
    )
    loss = loss_gte0 + loss_gte1
    return loss