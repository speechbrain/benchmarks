"""
Recipe  to train K-means clustering model on self-supervised representations.

To run this recipe, do the following:
> python train.py hparams/train_with_[SSL-model].yaml --data_folder=/path/to/LibriSPeech
Author
 * Pooneh Mousavi 2024
"""

import os
import sys
import logging
import random
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.kmeans import fetch_kmeans_model, train, save_model
import torchaudio
from speechbrain.utils.data_utils import download_file

logger = logging.getLogger(__name__)


def dataio_prepare(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data]

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    return train_data


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train"],
            "split_ratio": hparams["split_ratio"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load SSL model
    hparams["ssl_model"] = hparams["ssl_model"].to(run_opts["device"])

    # Make training Dataloader
    train_set = dataio_prepare(hparams)
    if not (
        isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)
    ):
        train_set = sb.dataio.dataloader.make_dataloader(
            train_set, **hparams["train_dataloader_opts"]
        )

    os.makedirs(hparams["save_folder"], exist_ok=True)
    # If you use dataloader checkpoints, make sure to keep all the settings as in the previous run and keep the dataset ordering the same.
    dataloader_path = os.path.join(
        hparams["save_folder"], "dataloader-TRAIN.ckpt"
    )
    if os.path.exists(dataloader_path):
        logger.info(
            f"The dataloader checkpoint is loaded from {dataloader_path}."
        )
        train_set._speechbrain_load(dataloader_path, False)

    # Load pretrained KMeans model if it exists. Otherwise,  create new one.
    checkpoint_path = os.path.join(
        hparams["save_folder"],
        f"kmeans-cluster-{hparams['num_clusters']}-layer-{hparams['ssl_layer_num']}.pt",
    )
    kmeans_model = fetch_kmeans_model(
        n_clusters=hparams["num_clusters"],
        init=hparams["init"],
        max_iter=hparams["max_iter"],
        batch_size=hparams["batch_size"],
        tol=hparams["tol"],
        max_no_improvement=hparams["max_no_improvement"],
        n_init=hparams["n_init"],
        reassignment_ratio=hparams["reassignment_ratio"],
        random_state=hparams["seed"],
        checkpoint_path=checkpoint_path,
    )

    # Train and save Kmeans model
    train(
        kmeans_model,
        train_set,
        hparams["ssl_model"],
        hparams["save_folder"],
        hparams["ssl_layer_num"],
        kmeans_batch_size=hparams["kmeans_batch_size"],
        device=run_opts["device"],
        checkpoint_interval=hparams["checkpoint_interval"],
    )

    logger.info(f"Saving kmeans model at {checkpoint_path}.")
    save_model(kmeans_model, checkpoint_path)
    train_set._speechbrain_save(dataloader_path)
