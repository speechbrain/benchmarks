import os
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
logger = logging.getLogger(__name__)


class Ultra_Brain(sb.Brain):
    def compute_forward(self, batch):
        print('START')
        batch = batch.to(self.device)
        rf = batch.sig
        a = self.modules.CnnBlock(rf)
        logits = self.modules.MLPBlock(a)
        
        print('OUT',logits)

        return logits 

    def compute_objectives(self, predictions, batch):
        return sb.nnet.losses.mse_loss(predictions, batch.att)

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
        sig, _, att = load_ultrasound(rf_data)
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

    train_data,  valid_data, test_data = dataio_prepare(hparams)
    
    Ultra_brain = Ultra_Brain(
    modules=hparams["modules"],
    opt_class=hparams["optim"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
    )

    Ultra_brain.fit(
    Ultra_brain.hparams.epoch_counter,
    train_data,
    valid_data,
    train_loader_kwargs=hparams["train_dataloader_opts"],
    valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )