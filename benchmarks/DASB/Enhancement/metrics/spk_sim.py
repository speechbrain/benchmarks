"""Cosine similarity between speaker embeddings.

Authors
* Luca Della Libera 2024
"""

import os

import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import AutoModelForAudioXVector


__all__ = ["SpkSimECAPATDNN", "SpkSimWavLM"]


SAMPLING_RATE = 16000


class ComputeScoreECAPATDNN:
    def __init__(self, model_hub, save_path, sampling_rate):
        self.model = SpeakerRecognition.from_hparams(
            model_hub, savedir=save_path
        )
        self.sampling_rate = sampling_rate

    def __call__(self, hyp_audio, ref_audio, sampling_rate):
        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, sampling_rate, self.sampling_rate
        )
        ref_audio = torchaudio.functional.resample(
            ref_audio, sampling_rate, self.sampling_rate
        )

        # Forward
        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)
        self.model.eval()
        score, _ = self.model.verify_batch(hyp_audio[None], ref_audio[None])
        return score.item()


class ComputeScoreWavLM:
    def __init__(self, model_hub, save_path, sampling_rate):
        self.model = AutoModelForAudioXVector.from_pretrained(
            model_hub, cache_dir=save_path
        )
        self.sampling_rate = sampling_rate

    def __call__(self, hyp_audio, ref_audio, sampling_rate):
        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, sampling_rate, self.sampling_rate
        )
        ref_audio = torchaudio.functional.resample(
            ref_audio, sampling_rate, self.sampling_rate
        )

        # Forward
        self.model.to(hyp_audio.device)
        self.model.eval()

        max_length = max(len(hyp_audio), len(ref_audio))
        attention_mask = torch.ones(
            2, max_length, dtype=torch.long, device=hyp_audio.device
        )
        attention_mask[0, len(hyp_audio) :] = 0
        attention_mask[1, len(ref_audio) :] = 0
        hyp_audio = torch.nn.functional.pad(
            hyp_audio, [0, max_length - len(hyp_audio)]
        )
        ref_audio = torch.nn.functional.pad(
            ref_audio, [0, max_length - len(ref_audio)]
        )

        hyp_embs, ref_embs = self.model(
            input_values=torch.stack([hyp_audio, ref_audio]),
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings

        score = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )
        return score.item()


root_folder = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(root_folder, "huggingface")

SpkSimECAPATDNN = ComputeScoreECAPATDNN(
    "speechbrain/spkrec-ecapa-voxceleb", save_path, SAMPLING_RATE
)
SpkSimWavLM = ComputeScoreWavLM(
    "microsoft/wavlm-base-sv", save_path, SAMPLING_RATE
)
