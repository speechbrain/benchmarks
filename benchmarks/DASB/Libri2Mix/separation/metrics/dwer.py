"""Differential WER (dWER) (see https://arxiv.org/abs/1911.07953).

Authors
 * Luca Della Libera 2024
"""

import torch
import torchaudio
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.lobes.models.huggingface_transformers import Whisper
from speechbrain.utils.metric_stats import ErrorRateStats, MetricStats


__all__ = ["DWER"]


SAMPLE_RATE = 16000


class DWER(MetricStats):
    def __init__(self, model_hub, save_path, sample_rate):
        self.sample_rate = sample_rate
        self.model = Whisper(
            model_hub, save_path, SAMPLE_RATE, freeze=True, freeze_encoder=True,
        ).cpu()
        self.searcher = S2SWhisperGreedySearcher(
            self.model, min_decode_ratio=0.0, max_decode_ratio=1.0,
        )
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.wer_computer = ErrorRateStats()

    def clear(self):
        self.wer_computer.clear()

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

        # Forward
        enc_out = self.model.forward_encoder(self.model._get_mel(audio))
        text, _, _, _ = self.searcher(enc_out, lens)
        text = self.model.tokenizer.batch_decode(text, skip_special_tokens=True)
        text = [self.model.tokenizer._normalize(x).split(" ") for x in text]
        hyp_text = text[: hyp_audio.shape[0]]
        ref_text = text[hyp_audio.shape[0] :]

        # Compute WER
        self.wer_computer.append(ids, hyp_text, ref_text)

    def summarize(self, field=None):
        return self.wer_computer.summarize(field)

    def write_stats(self, filestream, verbose=False):
        self.wer_computer.write_stats(filestream)
