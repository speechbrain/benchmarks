"""Deep Noise Suppression Mean Opinion Score (DNSMOS) (see https://arxiv.org/abs/2010.15258).

Authors
 * Luca Della Libera 2024
"""

# Adapted from:
# https://github.com/microsoft/DNS-Challenge/blob/4dfd2f639f737cdf530c61db91af16f7e0aa23e1/DNSMOS/dnsmos_local.py

import os

import librosa
import numpy as np
import onnxruntime as ort
import torchaudio
from speechbrain.utils.metric_stats import MetricStats


__all__ = ["DNSMOS"]


SAMPLE_RATE = 16000

INPUT_LENGTH = 9.01

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))

# PRIMARY_MODEL_PATH = os.path.join(ROOT_FOLDER, "sig_bak_ovr.onnx")

P808_MODEL_PATH = os.path.join(ROOT_FOLDER, "model_v8.onnx")


class DNSMOS(MetricStats):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        # self.onnx_sess = ort.InferenceSession(PRIMARY_MODEL_PATH)
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = os.cpu_count()
        sess_options.intra_op_num_threads = os.cpu_count()
        self.p808_onnx_sess = ort.InferenceSession(
            P808_MODEL_PATH, sess_options=sess_options
        )
        self.clear()

    def append(self, ids, hyp_audio, lens=None):
        assert hyp_audio.ndim == 2

        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, self.sample_rate, SAMPLE_RATE
        )
        hyp_audio = hyp_audio.cpu().numpy()

        # predicted_mos_sig_seg = []
        # predicted_mos_bak_seg = []
        # predicted_mos_ovr_seg = []
        # predicted_p808_mos = []
        for i, x in enumerate(hyp_audio):
            if lens is not None:
                abs_length = int(lens[i].cpu().numpy() * len(x))
                x = x[:abs_length]

            score = self._score(x)

            # Unused
            # predicted_mos_sig_seg.append(score[0])
            # predicted_mos_bak_seg.append(score[1])
            # predicted_mos_ovr_seg.append(score[2])

            # DNSMOS
            self.scores.append(score[3])

        self.ids += ids

    def _score(self, audio):
        fs = SAMPLE_RATE

        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs

        # predicted_mos_sig_seg = []
        # predicted_mos_bak_seg = []
        # predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + INPUT_LENGTH) * hop_len_samples
                )
            ]
            if len(audio_seg) < len_samples:
                continue

            # input_features = np.array(audio_seg).astype("float32")[None]
            p808_input_features = np.array(
                self._audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[None]
            # oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            # mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(
            #    None, oi
            # )[0][0]
            # mos_sig, mos_bak, mos_ovr = self._get_polyfit_val(
            #    mos_sig_raw, mos_bak_raw, mos_ovr_raw
            # )
            # predicted_mos_sig_seg.append(mos_sig)
            # predicted_mos_bak_seg.append(mos_bak)
            # redicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        # sig_mos = np.mean(predicted_mos_sig_seg)
        # bak_mos = np.mean(predicted_mos_bak_seg)
        # ovr_mos = np.mean(predicted_mos_ovr_seg)
        p808_mos = np.mean(predicted_p808_mos)

        return [], [], [], p808_mos

    def _audio_melspec(
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

    def _get_polyfit_val(self, sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly
