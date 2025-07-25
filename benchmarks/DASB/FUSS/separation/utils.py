"""Common utilities.

Authors
 * Luca Della Libera 2024
"""

import os

import speechbrain as sb
import torch
import torchaudio
from speechbrain.dataio.dataio import merge_csvs
from transformers.models.hubert.modeling_hubert import (
    HubertEncoderStableLayerNorm,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderStableLayerNorm,
)
from transformers.models.wavlm.modeling_wavlm import WavLMEncoderStableLayerNorm


__all__ = ["SBWav2Vec2ForwardWrapper", "dataio_prepare"]

CHUNK = 10.0


class EncodecHelper:
    def __init__(self, codec, device):
        self.codec = codec
        self.device = device

    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        self.codec.to(self.device).eval()
        toks, _ = self.codec.encode(sig.unsqueeze(1), lens)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        self.codec.to(self.device).eval()
        sig = self.codec.decode(toks)[:, 0]  # [B, T]
        return sig


class DacHelper:
    def __init__(self, codec, device, num_codebooks):
        self.codec = codec
        self.device = device
        self.num_codebooks = num_codebooks

    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        self.codec.to(self.device).eval()
        toks, _ = self.codec(
            sig[:, None], n_quantizers=self.num_codebooks
        )  # [B, K, N]
        toks = toks.movedim(-1, -2)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        self.codec.to(self.device).eval()
        qfeats, _, _ = self.codec.quantizer.from_codes(
            toks.movedim(-1, -2)
        )  # [B, K, N] -> [B, K, N]
        sig = self.codec.decode(qfeats)[:, 0]  # [B, T]
        return sig


class SQCodecHelper:
    def __init__(self, codec, device):
        self.codec = codec
        self.device = device

    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        # sig: [B, T]
        self.codec.to(self.device).eval()
        toks, _ = self.codec.encode(sig[:, None])  # [B, K * N]
        K = self.codec.n_codebook
        N = toks.shape[-1] // K
        toks = self._unflatten_codebooks(toks, N, K)  # [B, N, K]
        return toks

    def _flatten_codebooks(self, arr):
        assert (
            len(arr.shape) == 3
        ), "Input array must have 3 dimensions [B, N, K]"
        N, B, K = arr.shape
        arr = arr.clone()
        flattened_arr = arr.permute(1, 2, 0).reshape(B, N * K)
        return flattened_arr

    def _unflatten_codebooks(self, flat_arr, N, K):
        # flat_arr: [B, N * K]
        B = flat_arr.shape[0]
        return flat_arr.reshape(B, N, K)

    @torch.no_grad()
    def toks_to_sig(self, toks):
        toks = toks.permute(2, 0, 1)  # [B, N, K] -> [K, B, N]
        flat_toks = self._flatten_codebooks(toks).to(torch.int32)
        sig = self.codec.decode(flat_toks).squeeze(1)  # [B, T]
        return sig.to(toks.device)


class WavTokenizerHelper:
    def __init__(self, codec, device):
        self.codec = codec
        self.device = device

    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        self.codec.to(self.device).eval()
        toks, _ = self.codec.encode(sig)  # [B, K, N]
        toks = toks.permute(0, 2, 1)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        self.codec.to(self.device).eval()
        toks = toks.movedim(-1, -2)  # [B, N, K] -> [B, K, N]
        sig = self.codec.decode(toks)  # [B, T]
        return sig.clone()


class SBWav2Vec2ForwardWrapper(torch.nn.Module):
    """SpeechBrain wav2vec 2.0 wrapper that returns the hidden representations from the specified layer IDs.

    Arguments
    ---------
    wav2vec2:
        The SpeechBrain wav2vec 2.0 module.
    layer_ids:
        The layer IDs from which the hidden representations are extracted.

    Examples
    --------
    >>> import torch
    >>> from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    >>> from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    >>>
    >>> encoder = WavLM(source="microsoft/wavlm-large", save_path=HUGGINGFACE_HUB_CACHE)
    >>> encoder = SBWav2Vec2ForwardWrapper(encoder, layer_ids=[6, 7])
    >>>
    >>> input = torch.rand([10, 16000])
    >>> length = torch.ones(10)
    >>> output = encoder(input, length)

    """

    def __init__(self, wav2vec2, layer_ids):
        super().__init__()
        self.wav2vec2 = wav2vec2
        # Workaround to deal with hardcoded class name in discrete SSL
        # https://github.com/speechbrain/speechbrain/blob/60062c2536e8122253d6ad0e681208f554528950/speechbrain/lobes/models/huggingface_transformers/discrete_ssl.py#L88
        self.__class__.__name__ = self.wav2vec2.__class__.__name__
        self.layer_ids = sorted(layer_ids)
        assert hasattr(self.wav2vec2, "model")
        assert hasattr(self.wav2vec2.model, "encoder")
        assert hasattr(self.wav2vec2.model.encoder, "layers")
        # Workaround for early exiting to avoid the computational overhead of forwarding through the whole model
        # NOTE: the model is modified in-place
        self.wav2vec2.output_all_hiddens = True
        self.wav2vec2.model.encoder.layers = self.wav2vec2.model.encoder.layers[
            : max(self.layer_ids)
        ]
        # NOTE: workaround to account for layer norm applied to the last hidden states when StableLayerNorm variant is used:
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wavlm/modeling_wavlm.py#L816
        if isinstance(
            self.wav2vec2.model.encoder,
            (
                HubertEncoderStableLayerNorm,
                Wav2Vec2EncoderStableLayerNorm,
                WavLMEncoderStableLayerNorm,
            ),
        ):
            self.wav2vec2.model.encoder.layer_norm = torch.nn.Identity()

    def extract_features(self, wav, length=None):
        feats = self.wav2vec2(wav, length)  # (K, B, N, H)
        return feats

    def forward(self, wav, length=None):
        return self.extract_features(wav, length)


def dataio_prepare(
    data_folder,
    train_csv,
    valid_csv,
    test_csv,
    sample_rate=16000,
    train_remove_if_longer=60.0,
    valid_remove_if_longer=60.0,
    test_remove_if_longer=60.0,
    sorting="ascending",
    debug=False,
    **hparams,
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    """
    if isinstance(train_csv, (list, tuple)):
        csvs = [os.path.basename(x) for x in train_csv]
        save_folder = os.path.dirname(train_csv[0])
        merge_csvs(
            save_folder, csvs, "train.csv",
        )
        train_csv = os.path.join(save_folder, "train.csv")

    if isinstance(valid_csv, (list, tuple)):
        csvs = [os.path.basename(x) for x in valid_csv]
        save_folder = os.path.dirname(valid_csv[0])
        merge_csvs(
            save_folder, csvs, "valid.csv",
        )
        valid_csv = os.path.join(save_folder, "valid.csv")

    if isinstance(test_csv, (list, tuple)):
        csvs = [os.path.basename(x) for x in test_csv]
        save_folder = os.path.dirname(test_csv[0])
        merge_csvs(
            save_folder, csvs, "test.csv",
        )
        test_csv = os.path.join(save_folder, "test.csv")

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv, replacements={"DATA_ROOT": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=sorting == "descending",
        key_max_value={"duration": train_remove_if_longer},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_csv, replacements={"DATA_ROOT": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": valid_remove_if_longer},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=test_csv, replacements={"DATA_ROOT": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": test_remove_if_longer},
    )

    # train_data = train_data.overfit_test(32, 32)
    # valid_data = valid_data.overfit_test(8, 8)
    # test_data = test_data.overfit_test(3, 3)

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = [
        "mixture_wav",
        "background0_sound_wav",
        "foreground0_sound_wav",
        "foreground1_sound_wav",
        "foreground2_sound_wav",
    ]
    provides = ["in_sig", "out_sig"]

    def audio_pipeline(mix_wav, *src_wavs):
        # Mixed signal
        try:
            original_sample_rate = sb.dataio.dataio.read_audio_info(
                mix_wav
            ).sample_rate
            # total_frames = sb.dataio.dataio.read_audio_info(
            #     mix_wav
            # ).num_frames

            # start = randint(0, total_frames - int(CHUNK * original_sample_rate))

            # Source signals
            src_sigs = []
            for src_wav in src_wavs:
                assert (
                    original_sample_rate
                    == sb.dataio.dataio.read_audio_info(src_wav).sample_rate
                )
                src_sig = sb.dataio.dataio.read_audio(
                    dict(file=src_wav)
                )  # ,start=start, stop=start + int(CHUNK * original_sample_rate)))
                src_sigs.append(src_sig)
            src_sigs = torch.stack(src_sigs)  # [S, T]
            max_vals = torch.max(torch.abs(src_sigs), dim=1, keepdim=True)[
                0
            ]  # Find peak per source item
            src_sigs = torch.where(
                max_vals > 0, src_sigs / max_vals, src_sigs
            )  # Normalize only non-silent signals

            out_sig = torchaudio.functional.resample(
                src_sigs, original_sample_rate, sample_rate,
            )
            in_sig = out_sig.sum(0)  # [T]
        except Exception as e:
            print(e)
        yield in_sig

        # Flatten as SpeechBrain's dataloader does not support multichannel audio
        out_sig = out_sig.flatten()  # [S * T]
        yield out_sig

    sb.dataio.dataset.add_dynamic_item(
        [train_data, valid_data, test_data], audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import (
        Wav2Vec2,
    )

    for source in [
        "facebook/wav2vec2-large-960h-lv60-self",
        "facebook/hubert-large-ll60k",
        "microsoft/wavlm-large",
    ]:
        layer_ids = [3, 7]
        encoder1 = Wav2Vec2(
            source=source, save_path=HUGGINGFACE_HUB_CACHE, output_norm=True,
        )
        encoder1 = SBWav2Vec2ForwardWrapper(
            encoder1, layer_ids=layer_ids
        ).eval()

        encoder2 = Wav2Vec2(
            source=source,
            save_path=HUGGINGFACE_HUB_CACHE,
            output_norm=True,
            output_all_hiddens=True,
        ).eval()

        input = torch.ones([1, 16000])
        with torch.no_grad():
            output1 = encoder1(input)[layer_ids]
            output2 = encoder2(input)[layer_ids]

        print((output1 == output2).all())
