#!/usr/bin/env/python3
"""Recipe for training a Text-to-Speech system based on tokenized audio
Discrete SSL version

Inspired by WhisperSpeech
https://github.com/collabora/WhisperSpeech

However, this is not an implementation of WhisperSpeech, but rather
a radical simplification of it that uses only an acoustic model


Authors
 * Artem Ploujnikov 2024
"""

import torch
from train import TokotronBrain, run_experiment
from speechbrain.dataio.dataio import clean_padding_


class TokotronDiscreteSSLBrain(TokotronBrain):
    """Tokotron implementation for Encodec"""

    def on_stage_start(self, stage, epoch):
        self.compute_offset()
        return super().on_stage_start(stage, epoch)

    def compute_offset(self):
        """Computes per-layer offsets"""
        layers_set = set(self.hparams.token_model_layers)
        available_layers_set = set(self.hparams.vocoder_available_layers)
        if not layers_set.issubset(available_layers_set):
            unavailable_layers = ",".join(
                str(layer) for layer in (layers_set - available_layers_set)
            )
            raise ValueError(f"Layers {unavailable_layers} are not supported")
        self.num_units = self.hparams.audio_num_tokens
        _, layers_idx = torch.where(
            torch.tensor(
                self.hparams.vocoder_available_layers, device=self.device
            ).unsqueeze(0)
            == torch.tensor(
                self.hparams.token_model_layers, device=self.device
            ).unsqueeze(1)
        )
        self.layer_offset = (
            torch.tensor(layers_idx, device=self.device) * self.num_units
        )[None, None, :]
        self.offset = self.hparams.token_offset
        self.modules.vocoder.tokenize = False

    def create_waveform(self, audio, length):
        """Creates a waveform from a discrete or continuous audio
        representation

        Arguments
        ---------
        audio : torch.Tensor
            An audio tensor (Batch x Length x Heads or Batch x Length x Heads x Features)
        lengths : torch.Tensor
            A 1-D tensor

        Returns
        -------
        wav : torch.Tensor
        """
        units_with_offset = (
            audio + self.layer_offset.to(audio.device) + self.offset
        )
        wav = self.modules.vocoder(units_with_offset)
        wav = wav.squeeze(1)
        clean_padding_(wav, length)
        return wav


if __name__ == "__main__":
    run_experiment(TokotronDiscreteSSLBrain)
