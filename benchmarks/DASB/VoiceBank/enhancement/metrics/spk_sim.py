"""Cosine similarity between speaker embeddings.

Authors
 * Luca Della Libera 2024
"""

import torch
import torchaudio
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import MetricStats
from transformers import AutoModelForAudioXVector


__all__ = ["SpkSimECAPATDNN", "SpkSimWavLM"]


SAMPLE_RATE = 16000


class SpkSimECAPATDNN(MetricStats):
    def __init__(self, model_hub, save_path, sample_rate):
        self.sample_rate = sample_rate
        self.model = SpeakerRecognition.from_hparams(
            model_hub, savedir=save_path
        ).cpu()
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_audio, ref_audio, lens=None):
        assert hyp_audio.shape == ref_audio.shape
        assert hyp_audio.ndim == 2

        # Concatenate
        audio = torch.cat([hyp_audio, ref_audio])
        if lens is not None:
            lens = torch.cat([lens, lens])

        # Resample
        audio = torchaudio.functional.resample(
            audio, self.sample_rate, SAMPLE_RATE
        )

        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)
        self.model.eval()

        # Forward
        embs = self.model.encode_batch(audio, lens, normalize=False)
        hyp_embs, ref_embs = embs.split([len(hyp_audio), len(ref_audio)])
        scores = self.model.similarity(hyp_embs, ref_embs)[:, 0]

        self.ids += ids
        self.scores += scores.cpu().tolist()


class SpkSimWavLM(MetricStats):
    def __init__(self, model_hub, save_path, sample_rate):
        self.sample_rate = sample_rate
        self.model = AutoModelForAudioXVector.from_pretrained(
            model_hub, cache_dir=save_path
        )
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_audio, ref_audio, lens=None):
        assert hyp_audio.shape == ref_audio.shape
        assert hyp_audio.ndim == 2

        # Concatenate
        audio = torch.cat([hyp_audio, ref_audio])
        if lens is not None:
            lens = torch.cat([lens, lens])

        # Resample
        audio = torchaudio.functional.resample(
            audio, self.sample_rate, SAMPLE_RATE
        )

        self.model.to(hyp_audio.device)
        self.model.eval()

        # Attention mask
        attention_mask = None
        if lens is not None:
            abs_length = lens * audio.shape[-1]
            attention_mask = length_to_mask(
                abs_length.int()
            ).long()  # 0 for masked tokens

        # Forward
        embs = self.model(
            input_values=audio,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings

        hyp_embs, ref_embs = embs.split([len(hyp_audio), len(ref_audio)])
        scores = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )

        self.ids += ids
        self.scores += scores.cpu().tolist()
