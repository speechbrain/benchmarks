
import os
import numpy as np
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import torchaudio
from scipy.io import loadmat
from speechbrain.utils.parameter_transfer import Pretrainer
import matplotlib.pyplot as plt

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
    #print('Train Data', train_data)

    def load_ultrasound(ULTRA_PATH):
        dic = {}
        data_dic = loadmat(ULTRA_PATH)
        try:
            dic['rf_data'] = data_dic['rf_data'].reshape((-1,))
            dic['rf_env'] = data_dic['rf_env'].reshape((-1,))
            dic['my_att'] = data_dic['my_att'][0][0]
        except:
            dic['rf_data'] = data_dic['rf_data'].reshape((-1,))
            dic['my_att'] = data_dic['my_att'].item()
            dic['rf_env'] = 0
        return dic['rf_data'] , dic['rf_env'], dic['my_att']


    # 2. Define Ultrasound pipeline:
    @sb.utils.data_pipeline.takes("rf_data")
    @sb.utils.data_pipeline.provides("sig","att")
    def ultrasound_pipeline(rf_data):
        rf_data, _, att = load_ultrasound(rf_data)
        len_wav = rf_data.shape[0]
        pddd = 4500#4000

        if len_wav < pddd:
            pad = np.zeros(pddd - len_wav)
            rf_data = np.hstack([rf_data, pad])
        elif len_wav > pddd:
            rf_data = rf_data[:pddd]

        sig = rf_data
        #print('SIGNAL CALLINg from ultrasound pipline ',sig, att)
        yield sig
        yield att

    sb.dataio.dataset.add_dynamic_item(datasets, ultrasound_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["sig", "att",],)

    #print(valid_data[0])
    
    return (
        train_data,
        valid_data,
        test_data,
    )


def get_losses(log_file):
  """This function takes in input a path of a log-file and outputs the train and
  valid losses in lists of float numbers"""
  with open(log_file) as f:
      train_losses = []
      valid_losses =[]
      for line in f:
          if 'train loss' in line:
            train_loss = float(line.split('train loss: ')[1].split(' ')[0])
            train_losses.append(train_loss)
          if 'valid loss' in line:
            valid_loss = float(line.split('valid loss: ')[1])
            valid_losses.append(valid_loss)
      return train_losses, valid_losses
  


