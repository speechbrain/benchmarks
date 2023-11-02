#!/usr/bin/env/python3
"""Recipe for training an SSL-based ctc ASR system with librispeech.
 Decoding is performed with ctc greedy or LM-rescored decoder.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from tqdm.contrib import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader


logger = logging.getLogger(__name__)



def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )


    datasets = [train_data] 

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig"],
    )
    return train_data


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    hparams['ssl_model']= hparams['ssl_model'].to(run_opts['device'])

    checkpoint_path =  os.path.join(hparams['save_folder'] , f"kmeans_{hparams['n_clusters']}.pt")
    if  os.path.exists(checkpoint_path):
        kmeans = torch.load(checkpoint_path)
    else:
        # here we create the datasets objects as well as tokenization and encoding
        train_set = dataio_prepare(
            hparams
        )
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set =  sb.dataio.dataloader.make_dataloader(train_set, **hparams["train_dataloader_opts"])
        
        kmeans = MiniBatchKMeans(n_clusters=hparams['n_clusters'],
                            random_state=hparams['seed'],
                            batch_size=hparams['batch_size'],
                            n_init="auto")
        with tqdm(
                train_set,
                dynamic_ncols=True,
            ) as t:
                for batch in t:
                    batch = batch.to(run_opts['device'])
                    wavs, wav_lens = batch.sig
                    wavs, wav_lens = wavs.to(run_opts['device']), wav_lens.to(run_opts['device'])
                    feats = hparams['ssl_model'](wavs, wav_lens)
                    kmeans = kmeans.partial_fit(feats[hparams['ssl_layer_num']]) 
        torch.save(
            checkpoint_path,
            {
                "n_features_in_": kmeans.n_features_in_,
                "_n_threads": kmeans._n_threads,
                "cluster_centers_": kmeans.cluster_centers_,
            },
        )


   

