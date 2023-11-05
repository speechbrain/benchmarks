"""This lobe enables the integration of pretrained discrete Hubert.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Pooneh Mousavi 2023

"""
import logging
import torch
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT

logger = logging.getLogger(__name__)


class DiscreteHuBERT(HuBERT):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete HuBERT models.

    Source paper HuBERT: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    For now, HuggingFace's HuBERT and WavLM model can be loaded using the exact code for Wav2Vec2 model.
    For this reason, HuBERT and WavLM can be fine inheriting the Wav2Vec2 class.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/hubert-base-ls960"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the HuBERT model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the HuBERT model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface HubertModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example facebook/hubert-base-ls960 has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/hubert-base-ls960"
    >>> save_path = "savedir"
    >>> model = HuBERT(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        kmeans,
        output_norm=False,
        freeze=False,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        output_all_hiddens=True,
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            output_norm=output_norm,
            freeze=freeze,
            freeze_feature_extractor=freeze_feature_extractor,
            apply_spec_augment=apply_spec_augment,
            output_all_hiddens=output_all_hiddens,
        )

        self.kmean = kmeans

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        """

        # If we freeze, we simply remove all grads from the graph.
        feats = None
        if self.freeze:
            with torch.no_grad():
                feats = self.extract_features(wav, wav_lens)

        feats = self.extract_features(wav, wav_lens)

        x = self.kmeans.predict(feats.squeeze().cpu().numpy())
        return torch.tensor(x, dtype=torch.long, device=wav.device)
