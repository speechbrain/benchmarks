"""Deep Noise Suppression Mean Opinion Score (DNSMOS) (see https://arxiv.org/abs/2010.15258).

Authors
* Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/microsoft/DNS-Challenge/blob/4dfd2f639f737cdf530c61db91af16f7e0aa23e1/DNSMOS/dnsmos_local.py

import os

import librosa
import numpy as np
import onnxruntime as ort
import torchaudio


__all__ = ["DNSMOS"]


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    def __init__(self, p808_model_path, sampling_rate):
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        self.sampling_rate = sampling_rate

    def audio_melspec(
        self,
        audio,
        n_mels=120,
        frame_size=320,
        hop_length=160,
        sr=16000,
        to_db=True,
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def __call__(self, audio, sampling_rate):
        # Resample
        audio = torchaudio.functional.resample(
            audio, sampling_rate, self.sampling_rate
        )

        audio = audio.cpu().numpy()
        fs = self.sampling_rate

        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + INPUT_LENGTH) * hop_len_samples
                )
            ]
            if len(audio_seg) < len_samples:
                continue

            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[None]
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            predicted_p808_mos.append(p808_mos)

        return np.mean(predicted_p808_mos)


root_folder = os.path.dirname(os.path.realpath(__file__))
p808_model_path = os.path.join(root_folder, "model_v8.onnx")

DNSMOS = ComputeScore(p808_model_path, SAMPLING_RATE)
