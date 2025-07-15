"""Resynthesis metircs

Authors
 * Artem Ploujnikov 2024
"""

import csv
import torch
import torchaudio
from pathlib import Path
from torch import nn

from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.utils.fetching import fetch
from speechbrain.utils.metric_stats import MetricStats


UTMOS_SAMPLE_RATE = 16000
UTMOS_DEFAULT_JUDGE_ID = 288
UTMOS_DEFAULT_DOMAIN_ID = 0
UTMOS_DEFAULT_MODEL_NAME = "utmos.ckpt"


class UTMOSModel(nn.Module):
    """The UTMOS model wrapper

    Arguments
    ---------
    source : str
        The WavLM source
    save_path : str | path-like
        The path where the model will be saved
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    multiplier : float, optional
        The number that the raw model output is multiplied by
        to compute the score
    offset : float, optional
        The number that (raw output * multiplier) will be added
        to in order to get the score
    """

    def __init__(
        self,
        source,
        save_path,
        features_dim=768,
        num_domains=3,
        domain_dim=128,
        num_judges=3000,
        judge_dim=128,
        decoder_hidden_size=512,
        multiplier=2.0,
        offset=3.0,
    ):
        super().__init__()

        self.ssl_encoder = Wav2Vec2(
            source,
            save_path,
            freeze=True,
            output_norm=False,
            freeze_feature_extractor=True,
            output_all_hiddens=False,
        )

        self.domain_embedding = nn.Embedding(num_domains, domain_dim)
        self.judge_embedding = nn.Embedding(num_judges, judge_dim)

        self.decoder = nn.LSTM(
            input_size=features_dim + domain_dim + judge_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(decoder_hidden_size * 2, 2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        )
        self.multiplier = multiplier
        self.offset = offset

    def forward(self, wav, domain_id=None, judge_id=None):
        """Computes the forward pass

        Arguments
        ---------
        wav : torch.Tensor
            The raw waveforms
        domain_id : torch.Tensor
            The domain identifiers
        judge_id : torch.Tensor
            The judge identifier

        Returns
        -------
        result : torch.Tensor
            The predicted rating(s)
        """

        if domain_id is None:
            domain_id = torch.zeros(
                len(wav), dtype=torch.int, device=wav.device
            )
        if judge_id is None:
            judge_id = (
                torch.ones(len(wav), dtype=torch.int, device=wav.device)
                * UTMOS_DEFAULT_JUDGE_ID
            )

        ssl_features = self.ssl_encoder(wav)
        domain_emb = self.domain_embedding(domain_id)
        judge_emb = self.judge_embedding(judge_id)

        domain_emb = domain_emb.unsqueeze(1).expand(
            -1, ssl_features.size(1), -1
        )
        judge_emb = judge_emb.unsqueeze(1).expand(-1, ssl_features.size(1), -1)
        concatenated_feature = torch.cat(
            [ssl_features, domain_emb, judge_emb], dim=2
        )

        decoder_output, _ = self.decoder(concatenated_feature)
        pred = self.classifier(decoder_output)

        return pred.mean(dim=1).squeeze(1) * self.multiplier + self.offset


class UTMOSMetric(MetricStats):
    """A metric implementing UTMOS

    Arguments
    ---------
    sample_rate : int
        The audio sample rate
    source : str`, optional
        The HuggingFace hube name for the encoder
    save_path : str | path-like, optional
        The path where the model will be saved
    model_name : str, optional
        The name of the model
    model_url : str, optional
        The download URL for the model
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    domain_id : int, optional
        The domain identifier
    judge_id : int, optional
        The judge identifier
    run_opts : dict
        Run options when instantiating the metric
    """

    def __init__(
        self,
        sample_rate,
        source,
        save_path,
        model_name=None,
        model_url=None,
        features_dim=768,
        num_domains=3,
        domain_dim=128,
        num_judges=3000,
        judge_dim=128,
        decoder_hidden_size=512,
        domain_id=None,
        judge_id=None,
        run_opts=None,
    ):
        self.sample_rate = sample_rate
        self.clear()

        if model_name is None:
            model_name = UTMOS_DEFAULT_MODEL_NAME
        if domain_id is None:
            domain_id = UTMOS_DEFAULT_DOMAIN_ID
        if judge_id is None:
            judge_id = UTMOS_DEFAULT_JUDGE_ID
        if sample_rate is None:
            sample_rate = UTMOS_SAMPLE_RATE

        encoder_path = Path(save_path)
        self.model = UTMOSModel(
            source=source,
            save_path=encoder_path.as_posix(),
            features_dim=features_dim,
            num_domains=num_domains,
            domain_dim=domain_dim,
            num_judges=num_judges,
            judge_dim=judge_dim,
            decoder_hidden_size=decoder_hidden_size,
        )

        # Download utmos model checkpoint
        fetch(model_name, model_url, save_path)
        model_path = Path(save_path) / model_name
        assert model_path.exists()

        # Load weights
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.domain_id = domain_id
        self.judge_id = judge_id

        if run_opts:
            device = run_opts.get("device")
            if device:
                self.model.to(device)

    def append(
        self,
        ids,
        wavs,
        length=None,
        sample_rate=None,
        domain_ids=None,
        judge_ids=None,
        **kwargs,
    ):
        """Computes the UTMOS metric for the provided audio

        Arguments
        ---------
        ids : list
            The list of item IDs
        wavs : torch.Tensor
            The audio prediction to be evaluated (e.g. TTS output)
        length : torch.Tensor, optional
            Relative lengths
        sample_rate : int
            The sample rate
        domain_ids : torch.Tensor, optional
            The domain IDs. The default will be used if not provided
        judge_ids : torch.Tensor
            The judge IDs. The default will be used if not provided
        **kwargs: : dict
            Other arguments (ignored)
        """
        if wavs.dim() > 2:
            wavs = wavs.squeeze()

        # Resample
        hyp_audio = torchaudio.functional.resample(
            wavs, sample_rate, self.sample_rate
        )

        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)

        if domain_ids is None:
            domain_ids = torch.zeros(
                len(hyp_audio), dtype=torch.int, device=hyp_audio.device
            )
        if judge_ids is None:
            judge_ids = (
                torch.ones(
                    len(hyp_audio), dtype=torch.int, device=hyp_audio.device
                )
                * self.judge_id
            )

        output = self.model(hyp_audio, domain_ids, judge_ids)
        self.scores += output.cpu().tolist()

        self.ids += ids

    def summarize(self, field=None):
        """Returns a dict containing detailed UTMOS statistics. UTMOS
        itself produces only one score per utterance - but the summary
        will obtain full descriptive statistics (see `descriptive_statistics`)

        Arguments
        ---------
        field : str, optional
            The field to return, if you are only interested in one of them.
            If specified, a single `float` is returned, otherwise, a dict is.

        Returns
        -------
        dict from str to float, if `field is None`
            A dictionary of the fields documented above.
        float, if `field is not None`
            The single field selected by `field`.
        """
        stats = descriptive_statistics(self.scores, result_key="utmos")
        return stats[field] if field else stats

    def write_stats(self, filestream, verbose=False):
        writer = csv.writer(filestream)
        writer.writerow(["id", "utmos"])
        for uttid, row in zip(self.ids, self.scores):
            writer.writerow([uttid, row])


def descriptive_statistics(items, key=None, result_key=None):
    """Computes descriptive statistics for the summary

    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        The key of the metric for which the statistics will be computed
    result_key : str
        The key to use for results

    Returns
    -------
    statistics : dict
        The desccriptive statistics computed
            <result_key>_mean : the arithmetic mean
            <result_key>_std : the standard deviation
            <result_key>_min : the minimum value
            <result_key>_max : the maximum value
            <result_key>_median : the median value
            <result_key>_q1 : the first quartile
            <result_key>_q3 : the third quartile
            <result_key>_iqr : the interquartile ratio
    """
    if not items:
        return {}
    if not result_key:
        result_key = key
    if key is None:
        values = torch.tensor(items)
    else:
        values = torch.tensor([item[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{result_key}_{stat_key}": value.item()
        for stat_key, value in stats.items()
    }
