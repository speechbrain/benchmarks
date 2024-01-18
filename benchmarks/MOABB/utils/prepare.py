#!/usr/bin/python3
"""
Prepare MOABB datasets.

Author
------
Davide Borra, 2022
"""
import mne
import numpy as np
from moabb.paradigms import MotorImagery, P300, SSVEP
from mne.utils.config import set_config, get_config
import os
import pickle
from mne.channels import find_ch_adjacency
import scipy

# Set mne verbosity
mne.set_log_level(verbose="error")


def get_output_dict(
    dataset, subject, events_to_load, srate_in, srate_out, fmin, fmax, verbose=0
):
    """This function returns the dictionary with subject-specific data."""
    output_dict = {}
    output_dict["code"] = dataset.code
    output_dict["subject_list"] = dataset.subject_list
    output_dict["paradigm"] = dataset.paradigm
    output_dict["n_sessions"] = dataset.n_sessions
    output_dict["fmin"] = fmin
    output_dict["fmax"] = fmax
    output_dict["ival"] = dataset.interval
    output_dict["interval"] = list(
        np.array(dataset.interval) - np.min(dataset.interval)
    )
    output_dict["reference"] = dataset.doi

    event_id = dataset.event_id
    output_dict["event_id"] = event_id
    event_keys = list(event_id.keys())
    idx_sorting = []
    for e in event_keys:
        idx_sorting.append(int(event_id[e]) - 1)

    events = [event_keys[i] for i in np.argsort(idx_sorting)]
    if events_to_load is not None:
        events = [e for e in events if e in events_to_load]
    output_dict["events"] = events

    output_dict["original_srate"] = srate_in
    output_dict["srate"] = srate_out if srate_out is not None else srate_in

    paradigm = None
    if dataset.paradigm == "imagery":
        paradigm = MotorImagery(
            events=events_to_load,  # selecting all events or specific events
            n_classes=len(output_dict["events"]),  # setting number of classes
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out,  # downsample
        )
    elif dataset.paradigm == "p300":
        paradigm = P300(
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out,  # downsample
        )
    elif dataset.paradigm == "ssvep":
        paradigm = SSVEP(
            events=events_to_load,  # selecting all events or specific events
            n_classes=len(output_dict["events"]),  # setting number of classes
            fmin=fmin,  # band-pass filtering
            fmax=fmax,
            channels=None,  # all channels
            resample=srate_out,  # downsample
        )

    x, y, labels, metadata, channels, adjacency_mtx, srate = load_data(
        paradigm, dataset, [subject]
    )

    if verbose == 1:
        for l in np.unique(labels):
            print(
                print(
                    "Number of label {0} examples: {1}".format(
                        l, np.where(labels == l)[0].shape[0]
                    )
                )
            )

    if dataset.paradigm == "p300":
        if output_dict["events"] == ["Target", "NonTarget"]:
            y = 1 - y  # swap NonTarget to Target
            output_dict["events"] = ["NonTarget", "Target"]
    if verbose == 1:
        for c in np.unique(y):
            print(
                "Number of class {0} examples: {1}".format(
                    c, np.where(y == c)[0].shape[0]
                )
            )

    output_dict["channels"] = channels
    output_dict["adjacency_mtx"] = adjacency_mtx
    output_dict["x"] = x
    output_dict["y"] = y
    output_dict["labels"] = labels
    output_dict["metadata"] = metadata
    output_dict["subject"] = subject

    if verbose == 1:
        print(output_dict)
    return output_dict


def load_data(paradigm, dataset, idx):
    """This function returns EEG signals and the corresponding labels using MOABB methods
    In addition metadata, channel names and the sampling rate are provided too."""
    x, labels, metadata = paradigm.get_data(dataset, idx, True)
    ch_names = x.info.ch_names
    adjacency, _ = find_ch_adjacency(x.info, ch_type="eeg")
    adjacency_mtx = scipy.sparse.csr_matrix.toarray(
        adjacency
    )  # from sparse mtx to ndarray

    srate = x.info["sfreq"]
    x = x.get_data()
    y = [dataset.event_id[yy] for yy in labels]
    y = np.array(y)
    y -= y.min()
    return x, y, labels, metadata, ch_names, adjacency_mtx, srate


def download_data(data_folder, dataset):
    """This function download a specific MOABB dataset in a directory."""
    # changing default download directory
    mne_cfg = get_config()
    for a in mne_cfg.keys():
        if (
            mne_cfg[a] != data_folder
        ):  # reducing writes on mne cfg file to avoid conflicts in parallel trainings
            set_config(a, data_folder)
    dataset.download()


def prepare_data(
    data_folder,
    dataset,
    events_to_load,
    srate_in,
    srate_out,
    fmin,
    fmax,
    cached_data_folder=None,
    idx_subject_to_prepare=-1,
    save_prepared_dataset=True,
    verbose=0,
):
    """This function prepare all datasets and save them in a separate pickle for each subject."""

    # Crete the data folder (if needed)
    if not os.path.exists(data_folder):
        print(data_folder)
        os.makedirs(data_folder)

    # changing default download directory
    mne_cfg = get_config()
    for a in mne_cfg.keys():
        if (
            mne_cfg[a] != data_folder
        ):  # reducing writes on mne cfg file to avoid conflicts in parallel trainings
            set_config(a, data_folder)
    if cached_data_folder is None:
        cached_data_folder = data_folder
    tmp_output_dir = os.path.join(
        os.path.join(
            cached_data_folder,
            "MOABB_pickled",
            dataset.code,
            "{0}_{1}-{2}".format(
                str(
                    int(srate_out if srate_out is not None else srate_in)
                ).zfill(4),
                str(fmin).zfill(3),
                str(fmax).zfill(3),
            ),
        )
    )
    if not os.path.isdir(tmp_output_dir):
        os.makedirs(tmp_output_dir)

    if idx_subject_to_prepare < 0:
        subject_to_prepare = dataset.subject_list
    else:
        subject_to_prepare = [dataset.subject_list[idx_subject_to_prepare]]

    for kk, subject in enumerate(subject_to_prepare):
        fname = "sub-{0}.pkl".format(str(subject).zfill(3))
        output_dict_fpath = os.path.join(tmp_output_dir, fname)

        # Prepare dataset only if not already prepared
        output_dict = {}
        if os.path.isfile(output_dict_fpath):
            print("Using cached dataset at: {0}".format(output_dict_fpath))
            with open(output_dict_fpath, "rb") as handle:
                output_dict = pickle.load(handle)
        else:
            output_dict = get_output_dict(
                dataset,
                subject,
                events_to_load,
                srate_in,
                srate_out,
                fmin=fmin,
                fmax=fmax,
                verbose=verbose,
            )

        if save_prepared_dataset:
            if os.path.isfile(output_dict_fpath):
                print(
                    "Skipping data saving, a cached dataset was found at {0}".format(
                        output_dict_fpath
                    )
                )
            else:
                print("Saving the dataset at {0}".format(output_dict_fpath))
                with open(output_dict_fpath, "wb") as handle:
                    pickle.dump(
                        output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
        if (
            idx_subject_to_prepare > -1
        ):  # iterating over only 1 subject, return its dictionary
            return output_dict
