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

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["clean_wav", "noisy_wav"]
    provides = ["in_sig", "out_sig"]

    def audio_pipeline(clean_wav, noisy_wav):
        # Clean signal
        original_sample_rate = sb.dataio.dataio.read_audio_info(
            clean_wav
        ).sample_rate
        clean_sig = sb.dataio.dataio.read_audio(clean_wav)

        # Noisy signal
        assert (
            original_sample_rate
            == sb.dataio.dataio.read_audio_info(noisy_wav).sample_rate
        )
        noisy_sig = sb.dataio.dataio.read_audio(noisy_wav)

        in_sig = torchaudio.functional.resample(
            noisy_sig, original_sample_rate, sample_rate,
        )
        yield in_sig

        out_sig = torchaudio.functional.resample(
            clean_sig, original_sample_rate, sample_rate,
        )
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
