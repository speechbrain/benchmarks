import math
import logging
import pathlib as pl
import kaldiio
import torch
import numpy as np
from tqdm.auto import tqdm
import speechbrain as sb
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset


logger = logging.getLogger(__name__)


def get_device(use_cuda):
    logger.info("=" * 30)
    logger.info(f"USE_CUDA SET TO: {use_cuda}")
    logger.info(f"CUDA AVAILABLE?: {torch.cuda.is_available()}")
    logger.info("=" * 30)
    use_cuda = use_cuda and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


class TokensExtractor:
    """
    Extracts tokens from audio data using a tokenizer and saves them to a specified format.

    Arguments
    ---------
    tokenizer : torch.nn.Module
        The tokenizer model to use for token extraction.
    save_path : str
        The directory where the tokens will be saved.
    src_key : str, optional
        The key in the dataset that contains the audio data (default: "wav").
    id_key : str, optional
        The key in the dataset that contains unique identifiers (default: "id").
    save_format : str, optional
        The format to save the tokens ('numpy', 'pickle', 'soundfile_flac') (default: "numpy").
    use_cuda : bool, optional
        Whether to use CUDA for computation (default: True).
    dataloader_opts : dict, optional
        Options for the data loader (default: None).
    pipelines : list, optional
        List of data processing pipelines to apply (default: None).
    save_name : str, optional
        Base name for the saved token files (default: "tokens").
    """

    def __init__(
        self,
        tokenizer,
        save_path,
        src_key="wav",
        id_key="id",
        save_format="numpy",
        use_cuda=True,
        dataloader_opts=None,
        pipelines=None,
        save_name="tokens",
    ):
        """
        Initializes the TokensExtractor.

        Arguments
        ---------
        tokenizer : torch.nn.Module
            The tokenizer model to use for token extraction.
        save_path : str
            The directory where the tokens will be saved.
        src_key : str, optional
            The key in the dataset that contains the audio data (default: "wav").
        id_key : str, optional
            The key in the dataset that contains unique identifiers (default: "id").
        save_format : str, optional
            The format to save the tokens ('numpy', 'pickle', 'soundfile_flac') (default: "numpy").
        use_cuda : bool, optional
            Whether to use CUDA for computation (default: True).
        dataloader_opts : dict, optional
            Options for the data loader (default: None).
        pipelines : list, optional
            List of data processing pipelines to apply (default: None).
        save_name : str, optional
            Base name for the saved token files (default: "tokens").

        Raises
        ------
        ValueError
            If an unsupported save_format is provided.
        """
        self.save_path = pl.Path(save_path).absolute()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_name = save_name

        self.id_key = id_key
        self.src_key = src_key

        self.device = get_device(use_cuda)
        self.tokenizer = tokenizer.to(self.device)

        if save_format not in ["numpy", "pickle", "soundfile_flac"]:
            raise ValueError(f"Unsupported save_format: {save_format}")
        self.save_format = save_format

        if not dataloader_opts:
            dataloader_opts = {}
        self.dataloader_opts = dataloader_opts
        self.pipelines = pipelines if pipelines is not None else []

        self.wspecifier = f"ark,scp,t:{self.save_path}/{self.save_name}.ark,{self.save_path}/{self.save_name}.scp"
        self.writer = kaldiio.WriteHelper(
            self.wspecifier, write_function="numpy"
        )

    def extract(self, dataset):
        """
        Extracts tokens from the dataset and saves them to the specified format.

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset or dict
            The dataset from which to extract tokens. Can be a DynamicItemDataset or a dictionary.
        """
        if isinstance(dataset, dict):
            dataset = DynamicItemDataset(dataset)
        dataset.set_output_keys([self.src_key, self.id_key])
        for pipeline in self.pipelines:
            dataset.add_dynamic_item(pipeline)

        dataloader = make_dataloader(dataset, **self.dataloader_opts)
        batch_size = self.dataloader_opts.get("batch_size", 1)
        batch_count = int(math.ceil(len(dataset) / batch_size))
        for batch in tqdm(dataloader, total=batch_count):
            batch = batch.to(self.device)
            x, x_lengths = batch[self.src_key]
            ids = batch[self.id_key]
            batch_tokens = self.tokenizer.sig_to_tokens(x, x_lengths)
            batch_tokens = sb.utils.data_utils.undo_padding(
                batch_tokens, x_lengths
            )
            self.process_batch(batch_tokens, ids)

    def process_batch(self, batch, ids):
        """
        Processes a batch of tokens and writes them to the output files.

        Arguments
        ---------
        batch : list
            A list of tokens for each item in the batch.
        ids : list
            A list of unique identifiers corresponding to each item in the batch.
        """
        for tokens, utt_id in zip(batch, ids):
            tokens = np.array(tokens)
            self.writer(utt_id, tokens)

    def __del__(self):
        """
        Close the writer.
        """
        self.writer.close()


class TokensLoader:
    """
    A loader class for retrieving tokens corresponding to utterance IDs.

    Arguments
    ---------
    data_path: str
        The path to the data directory containing the token files.
    save_name: str, optional
        The base name of the tokens files (default: "tokens").
    """

    def __init__(
        self,
        data_path,
        save_name="tokens",
    ):
        """
        Initializes the TokensLoader.

        Arguments
        ---------
        data_path: str
            The path to the data directory containing the token files.
        save_name: str, optional
            The base name of the tokens files (default: "tokens").
        """
        self.data_path = pl.Path(data_path)
        if not self.data_path.exists():
            raise ValueError(
                f"Data folder not found: {self.data_path.as_posix()}"
            )
        self.tokens = self._load(data_path, save_name)

    def tokens_by_uttid(self, utt_id):
        """
        Retrieves the tokens corresponding to a given utterance ID.

        Arguments
        ---------
        utt_id: str
            The utterance ID to retrieve tokens for.

        Returns
        -------
        result: torch.LongTensor [T, N_Q]
            The tokens associated with the utterance ID.

        Raises
        ------
        KeyError
            If the utterance ID is not found in the tokens.
        """
        if utt_id not in self.tokens:
            raise KeyError(f"Utterance ID '{utt_id}' not found in tokens.")
        tokens_path = self.tokens[utt_id]
        tokens = kaldiio.load_mat(tokens_path)
        tokens = torch.from_numpy(tokens).long()
        return tokens

    def _load(self, data_path, save_name):
        """
        Loads the mapping from utterance IDs to token file paths.

        Arguments
        ---------
        data_path: str
            The path to the data directory containing the token files.
        save_name: str
            The base name of the tokens files.

        Returns
        -------
        utt2toks: dict
            A dictionary mapping utterance IDs to their corresponding token file paths.
        """
        scp_path = f"{data_path}/{save_name}.scp"
        with open(scp_path, "r") as f:
            utt2toks = {
                line.strip().split(None, 1)[0]: line.strip().split(None, 1)[1]
                for line in f
                if line.strip()
            }
        return utt2toks
