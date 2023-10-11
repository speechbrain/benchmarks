import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import torchaudio
from speechbrain.utils.parameter_transfer import Pretrainer
logger = logging.getLogger(__name__)



def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
    )
    if hparams["sorting"] == "ascending":
        # sorting data based on Attenuation!
        train_data = train_data.filtered_sorted(sort_key="attenuation")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="attenuation", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "bebe! sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
    )
    valid_data = valid_data.filtered_sorted(sort_key="attenuation")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"],
    )
    test_data = test_data.filtered_sorted(sort_key="attenuation")


    datasets = [train_data, valid_data, test_data]
    print('Train Data', train_data)
    # # We get the tokenizer as we need it to encode the labels when creating
    # # mini-batches.
    # tokenizer_sem = hparams["tokenizer_semantics"]
    # tokenizer_tr = hparams["tokenizer_transcription"]
    # # 2. Define audio pipeline:
    # @sb.utils.data_pipeline.takes("semantics")
    # @sb.utils.data_pipeline.provides("semantics", "tokens_list_sm",
    #                                  "tokens_bos_sm", "tokens_eos_sm", "tokens_sm")
    # def in_text_pipeline(semantics):
    #     yield semantics
    #     tokens_list_sm = tokenizer_sem.encode_as_ids(semantics)
    #     yield tokens_list_sm
    #     tokens_bos_sm = torch.LongTensor([hparams["bos_index"]] + (tokens_list_sm))
    #     yield tokens_bos_sm
    #     tokens_eos_sm = torch.LongTensor(tokens_list_sm + [hparams["eos_index"]])
    #     yield tokens_eos_sm
    #     tokens_sm = torch.LongTensor(tokens_list_sm)
    #     yield tokens_sm

    # sb.dataio.dataset.add_dynamic_item(datasets, in_text_pipeline)
    # #import sentencepiece

    # # 3. Define text pipeline:
    # @sb.utils.data_pipeline.takes("transcription")
    # @sb.utils.data_pipeline.provides(
    #     "transcription", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    # )
    # def text_pipeline(transcription):
    #     #tokenizer = sentencepiece.SentencePieceProcessor(
    #     #    model_file=hparams['tokenizer'])
        
    #     yield transcription
    #     tokens_list = tokenizer_tr.encode_as_ids(transcription)
    #     yield tokens_list
    #     tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
    #     yield tokens_bos
    #     tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
    #     yield tokens_eos
    #     tokens = torch.LongTensor(tokens_list)
    #     yield tokens

    # sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # # 4. Set output:
    # sb.dataio.dataset.set_output_keys(
    #     datasets, ["id", "semantics", "transcription",
    #                "tokens_bos", "tokens_eos", "tokens",
    #                "tokens_list_sm", "tokens_bos_sm",
    #                "tokens_eos_sm", "tokens_sm"],
    # )

    # return (
    #     train_data,
    #     valid_data,
    #     test_data,
    # )





if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    print(hparams_file, run_opts, overrides)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        print(hparams)
    
    sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
    )

    dataio_prepare(hparams)