"""
Data Iterators for MOABB Datasets and Paradigms

This module provides various data iterators tailored for MOABB datasets and paradigms.
Different training strategies have been implemented as distinct data iterators, including:
- Leave-One-Session-Out
- Leave-One-Subject-Out

Author
------
Davide Borra, 2021
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from utils.prepare import prepare_data


def get_idx_train_valid_classbalanced(idx_train, valid_ratio, y):
    """This function returns train and valid indices balanced across classes."""
    idx_train = np.array(idx_train)
    nclasses = y[idx_train].max() + 1

    idx_valid = []
    for c in range(nclasses):
        to_select_c = idx_train[np.where(y[idx_train] == c)[0]]
        # fixed validation examples equally spaced within recording
        idx = np.linspace(
            0,
            to_select_c.shape[0] - 1,
            round(valid_ratio * to_select_c.shape[0]),
        )
        idx = np.floor(idx).astype(int)
        tmp_idx_valid_c = to_select_c[idx]
        idx_valid.extend(tmp_idx_valid_c)
    print("Validation indices: {0}".format(idx_valid))
    idx_valid = np.array(idx_valid)
    idx_train = np.setdiff1d(idx_train, idx_valid)
    return idx_train, idx_valid


def get_dataloader(batch_size, xy_train, xy_valid, xy_test):
    """This function returns dataloaders for training, validation and test"""
    x_train, y_train = xy_train[0], xy_train[1]
    x_valid, y_valid = xy_valid[0], xy_valid[1]
    x_test, y_test = xy_test[0], xy_test[1]

    inps = torch.Tensor(
        x_train.reshape(
            (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        )
    )
    tgts = torch.tensor(y_train, dtype=torch.long)
    ds = TensorDataset(inps, tgts)
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    inps = torch.Tensor(
        x_valid.reshape(
            (x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
        )
    )
    tgts = torch.tensor(y_valid, dtype=torch.long)
    ds = TensorDataset(inps, tgts)
    valid_loader = DataLoader(ds, batch_size=batch_size, pin_memory=True)

    inps = torch.Tensor(
        x_test.reshape(
            (
                x_test.shape[0],
                x_test.shape[1],
                x_test.shape[2],
                1,
            )
        )
    )
    tgts = torch.tensor(y_test, dtype=torch.long)
    ds = TensorDataset(inps, tgts)
    test_loader = DataLoader(ds, batch_size=batch_size, pin_memory=True)

    return train_loader, valid_loader, test_loader


def crop_signals(x, srate, interval_in, interval_out):
    """Function that crops signals within a fixed window"""
    time = np.arange(interval_in[0], interval_in[1], 1 / srate)
    idx_start = np.argmin(np.abs(time - interval_out[0]))
    idx_stop = np.argmin(np.abs(time - interval_out[1]))
    return x[..., idx_start:idx_stop]


def get_neighbour_channels(
    adjacency_mtx, ch_names, n_steps=1, seed_nodes=["Cz"]
):
    """Function that samples a subset of channels from a seed channel including neighbour channels within a fixed number of steps in the adjacency matrix."""
    sel_channels = []
    for i in np.arange(n_steps):
        tmp_sel_channels = []
        for node in seed_nodes:
            idx_node = np.where(node == np.array(ch_names))[0][0]
            idx_linked_nodes = np.where(adjacency_mtx[idx_node, :] > 0)[
                0
            ]  # find indices linked to the node
            linked_channels = np.array(ch_names)[idx_linked_nodes]
            tmp_sel_channels.extend(list(linked_channels))
        seed_nodes = tmp_sel_channels
        sel_channels.extend(tmp_sel_channels)
    sel_channels = np.unique(sel_channels)
    return sel_channels


def sample_channels(x, adjacency_mtx, ch_names, n_steps, seed_nodes=["Cz"]):
    """Function that select only selected channels from the input data"""
    sel_channels = get_neighbour_channels(
        adjacency_mtx, ch_names, n_steps=n_steps, seed_nodes=seed_nodes
    )
    sel_channels = list(sel_channels)
    idx_sel_channels = []
    for k, ch in enumerate(ch_names):
        if ch in sel_channels:
            idx_sel_channels.append(k)
    idx_sel_channels = np.array(idx_sel_channels)  # idx selected channels

    if idx_sel_channels.shape[0] != x.shape[1]:
        x = x[:, idx_sel_channels, :]
        sel_channels_ = np.array(ch_names)[idx_sel_channels]
        sel_channels_ = list(sel_channels_)
        # should correspond to sel_channels ordered based on ch_names ordering
        print("Sampling channels: {0}".format(sel_channels_))
    else:
        print("Sampling all channels available: {0}".format(ch_names))
    return x


class LeaveOneSessionOut(object):
    """Leave one session out iterator for MOABB datasets.
    Designing within-subject, cross-session and session-agnostic iterations on the dataset for a specific paradigm.
    For each subject, one session is held back as test set and the remaining ones are used to train neural networks.
    The validation set can be sampled from a separate (and held back) session if enough sessions are available; otherwise, the validation set is sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    seed: int
        Seed for random number generators.
    """

    def __init__(self, seed):
        self.iterator_tag = "leave-one-session-out"
        np.random.seed(seed)

    def prepare(
        self,
        data_folder=None,
        cached_data_folder=None,
        dataset=None,
        batch_size=None,
        valid_ratio=None,
        target_subject_idx=None,
        target_session_idx=None,
        events_to_load=None,
        original_sample_rate=None,
        sample_rate=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        save_prepared_dataset=None,
        n_steps_channel_selection=None,
        seed_nodes=["Cz"],
    ):
        """This function returns the pre-processed datasets (training, validation and test sets).

        Arguments
        ---------
        data_folder: str
            String containing the path where data were downloaded.
        cached_data_folder: str
            String containing the path where data will be cached (into .pkl files) during preparation.
            This is convenient to speed up overall training when performing multiple trainings on EEG signals
            pre-processed always in the same way.
        dataset: moabb.datasets.?
            MOABB dataset.
        batch_size: int
            Mini-batch size.
        valid_ratio: float
            Ratio for extracting the validation set from the available training set (between 0 and 1).
        target_subject_idx: int
            Index of the subject to use to train the network.
        target_session_idx: int
            Index of the session to use to train the network.
        events_to_load: list
            List of 'events' considered when loading the MOABB dataset.
            It serves to load specific conditions (e.g., ["right_hand", "left_hand"] for specific movement conditions).
            See MOABB documentation and reference publications of each dataset for additional details about datasets.
        original_sample_rate: int
            Sampling rate of the loaded dataset (Hz).
        sample_rate: int
            Target sampling rate (Hz).
        fmin: float
            Low cut-off frequency of the applied band-pass filtering (Hz).
        fmax: float
            High cut-off frequency of the applied band-pass filtering (Hz).
        tmin: float
            Start time of the EEG epoch, with respect to the event as defined in the dataset (s).
            See MOABB documentation and reference publications of each dataset for additional details about datasets.
        tmax: float
            Stop time of the EEG epoch, with respect to the event as defined in the dataset (s).
            See MOABB documentation and reference publications of each dataset for additional details about datasets.
        save_prepared_dataset: bool
            Flag to save the prepared dataset into a pkl file.
        n_steps_channel_selection: int
            Number of steps to perform when sampling a subset of channels from a seed channel, based on the adjacency matrix.

        Returns
        -------
        tail_path: str
            String containing the relative path where results will be stored for the specified iterator, subject and session.
        datasets: dict
            Dictionary containing all sets (keys: 'train', 'test', 'valid').
        """
        interval = [tmin, tmax]

        # preparing or loading dataset
        data_dict = prepare_data(
            data_folder=data_folder,
            cached_data_folder=cached_data_folder,
            dataset=dataset,
            events_to_load=events_to_load,
            srate_in=original_sample_rate,
            srate_out=sample_rate,
            fmin=fmin,
            fmax=fmax,
            idx_subject_to_prepare=target_subject_idx,
            save_prepared_dataset=save_prepared_dataset,
        )

        x = data_dict["x"]
        y = data_dict["y"]
        srate = data_dict["srate"]
        original_interval = data_dict["interval"]
        metadata = data_dict["metadata"]
        if np.unique(metadata.session).shape[0] < 2:
            raise (
                ValueError(
                    "The number of sessions in the dataset must be >= 2 for leave-one-session-out iterations"
                )
            )
        sessions = np.unique(metadata.session)
        sess_id_test = [sessions[target_session_idx]]
        sess_id_train = np.setdiff1d(sessions, sess_id_test)
        sess_id_train = list(sess_id_train)
        print(
            "Session/sessions used as training and validation set: {0}".format(
                sess_id_train
            )
        )
        print("Session used as test set: {0}".format(sess_id_test))
        # iterate over sessions to accumulate session train and valid examples in a balanced way across sessions
        idx_train, idx_valid = [], []
        for s in sess_id_train:
            # obtaining indices for the current session
            idx = np.where(metadata.session == s)[0]
            # validation set definition (equal proportion btw classes)
            (tmp_idx_train, tmp_idx_valid) = get_idx_train_valid_classbalanced(
                idx, valid_ratio, y
            )
            idx_train.extend(tmp_idx_train)
            idx_valid.extend(tmp_idx_valid)

        idx_test = []
        for s in sess_id_test:
            # obtaining indices for the current session
            idx = np.where(metadata.session == s)[0]
            idx_test.extend(idx)

        idx_train = np.array(idx_train)
        idx_valid = np.array(idx_valid)
        idx_test = np.array(idx_test)

        x_train = x[idx_train, ...]
        y_train = y[idx_train]
        x_valid = x[idx_valid, ...]
        y_valid = y[idx_valid]
        x_test = x[idx_test, ...]
        y_test = y[idx_test]

        # time cropping
        if interval != original_interval:
            x_train = crop_signals(
                x=x_train,
                srate=srate,
                interval_in=original_interval,
                interval_out=interval,
            )
            x_valid = crop_signals(
                x=x_valid,
                srate=srate,
                interval_in=original_interval,
                interval_out=interval,
            )
            x_test = crop_signals(
                x=x_test,
                srate=srate,
                interval_in=original_interval,
                interval_out=interval,
            )

        # channel sampling
        if n_steps_channel_selection is not None:
            x_train = sample_channels(
                x_train,
                data_dict["adjacency_mtx"],
                data_dict["channels"],
                n_steps=n_steps_channel_selection,
                seed_nodes=seed_nodes,
            )
            x_valid = sample_channels(
                x_valid,
                data_dict["adjacency_mtx"],
                data_dict["channels"],
                n_steps=n_steps_channel_selection,
                seed_nodes=seed_nodes,
            )
            x_test = sample_channels(
                x_test,
                data_dict["adjacency_mtx"],
                data_dict["channels"],
                n_steps=n_steps_channel_selection,
                seed_nodes=seed_nodes,
            )

        # swap axes: from (N_examples, C, T) to (N_examples, T, C)
        x_train = np.swapaxes(x_train, -1, -2)
        x_valid = np.swapaxes(x_valid, -1, -2)
        x_test = np.swapaxes(x_test, -1, -2)

        # dataloaders
        train_loader, valid_loader, test_loader = get_dataloader(
            batch_size, (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        )
        datasets = {}
        datasets["train"] = train_loader
        datasets["valid"] = valid_loader
        datasets["test"] = test_loader
        tail_path = os.path.join(
            self.iterator_tag,
            "sub-{0}".format(
                str(dataset.subject_list[target_subject_idx]).zfill(3)
            ),
            sessions[target_session_idx],
        )
        return tail_path, datasets


class LeaveOneSubjectOut(object):
    """Leave one subject out iterator for MOABB datasets.
    Designing cross-subject, cross-session and subject-agnostic iterations on the dataset for a specific paradigm.
    One subject is held back as test set and the remaining ones are used to train neural networks.
    The validation set is sampled from the training set.
    All sets are extracted balanced across subjects, sessions and classes.

    Arguments
    ---------
    seed: int
        Seed for random number generators.
    """

    def __init__(self, seed):
        self.iterator_tag = "leave-one-subject-out"
        np.random.seed(seed)

    def prepare(
        self,
        data_folder=None,
        cached_data_folder=None,
        dataset=None,
        batch_size=None,
        valid_ratio=None,
        target_subject_idx=None,
        target_session_idx=None,
        events_to_load=None,
        original_sample_rate=None,
        sample_rate=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        save_prepared_dataset=None,
        n_steps_channel_selection=None,
        seed_nodes=["Cz"],
    ):
        """This function returns the pre-processed datasets (training, validation and test sets).

        Arguments
        ---------
        data_folder: str
            String containing the path where data were downloaded.
        cached_data_folder: str
            String containing the path where data will be cached (into .pkl files) during preparation.
            This is convenient to speed up overall training when performing multiple trainings on EEG signals
            pre-processed always in the same way.
        dataset: moabb.datasets.?
            MOABB dataset.
        batch_size: int
            Mini-batch size.
        valid_ratio: float
            Ratio for extracting the validation set from the available training set (between 0 and 1).
        target_subject_idx: int
            Index of the subject to use to train the network.
        target_session_idx: int
            Index of the session to use to train the network.
            For leave-one-subject-out this parameter is ininfluent (it is not used).
            It was held for symmetry with leave-one-session-out iterator and for convenience with the rest of the code.
        events_to_load: list
            List of 'events' considered when loading the MOABB dataset.
            It serves to load specific conditions (e.g., ["right_hand", "left_hand"] for specific movement conditions).
            See MOABB documentation and reference publications of each dataset for additional details about datasets.
        original_sample_rate: int
            Sampling rate of the loaded dataset (Hz).
        sample_rate: int
            Target sampling rate (Hz).
        fmin: float
            Low cut-off frequency of the applied band-pass filtering (Hz).
        fmax: float
            High cut-off frequency of the applied band-pass filtering (Hz).
        tmin: float
            Start time of the EEG epoch, with respect to the event as defined in the dataset (s).
            See MOABB documentation and reference publications of each dataset for additional details about datasets.
        tmax: float
            Stop time of the EEG epoch, with respect to the event as defined in the dataset (s).
            See MOABB documentation and reference publications of each dataset for additional details about datasets.
        save_prepared_dataset: bool
            Flag to save the prepared dataset into a pkl file.
        n_steps_channel_selection: int
            Number of steps to perform when sampling a subset of channels from a seed channel, based on the adjacency matrix.

        Returns
        -------
        tail_path: str
            String containing the relative path where results will be stored for the specified iterator, subject and session.
        datasets: dict
            Dictionary containing all sets (keys: 'train', 'test', 'valid').
        """
        interval = [tmin, tmax]
        if len(dataset.subject_list) < 2:
            raise (
                ValueError(
                    "The number of subjects in the dataset must be >= 2 for leave-one-subject-out iterations"
                )
            )

        # preparing or loading test set
        data_dict = prepare_data(
            data_folder=data_folder,
            cached_data_folder=cached_data_folder,
            dataset=dataset,
            events_to_load=events_to_load,
            srate_in=original_sample_rate,
            srate_out=sample_rate,
            fmin=fmin,
            fmax=fmax,
            idx_subject_to_prepare=target_subject_idx,
            save_prepared_dataset=save_prepared_dataset,
        )

        x_test = data_dict["x"]
        y_test = data_dict["y"]
        original_interval = data_dict["interval"]
        srate = data_dict["srate"]

        subject_idx_train = [
            i
            for i in np.arange(len(dataset.subject_list))
            if i != target_subject_idx
        ]
        subject_ids_train = list(
            np.array(dataset.subject_list)[np.array(subject_idx_train)]
        )

        print(
            "Subject/subjects used as training and validation set: {0}".format(
                subject_ids_train
            )
        )
        print(
            "Subject used as test set: {0}".format(
                dataset.subject_list[target_subject_idx]
            )
        )

        x_train, y_train, x_valid, y_valid = [], [], [], []
        for subject_idx in subject_idx_train:
            # preparing or loading training/valid set
            data_dict = prepare_data(
                data_folder=data_folder,
                cached_data_folder=cached_data_folder,
                dataset=dataset,
                events_to_load=events_to_load,
                srate_in=original_sample_rate,
                srate_out=sample_rate,
                fmin=fmin,
                fmax=fmax,
                idx_subject_to_prepare=subject_idx,
                save_prepared_dataset=save_prepared_dataset,
            )

            tmp_x_train = data_dict["x"]
            tmp_y_train = data_dict["y"]
            tmp_metadata = data_dict["metadata"]

            # defining training and validation indices from subjects and sessions in a balanced way
            idx_train, idx_valid = [], []
            for session in np.unique(tmp_metadata.session):
                idx = np.where(tmp_metadata.session == session)[0]
                # validation set definition (equal proportion btw classes)
                (
                    tmp_idx_train,
                    tmp_idx_valid,
                ) = get_idx_train_valid_classbalanced(
                    idx, valid_ratio, tmp_y_train
                )
                idx_train.extend(tmp_idx_train)
                idx_valid.extend(tmp_idx_valid)
            idx_train = np.array(idx_train)
            idx_valid = np.array(idx_valid)

            tmp_x_valid = tmp_x_train[idx_valid, ...]
            tmp_y_valid = tmp_y_train[idx_valid]
            tmp_x_train = tmp_x_train[idx_train, ...]
            tmp_y_train = tmp_y_train[idx_train]

            x_train.extend(tmp_x_train)
            y_train.extend(tmp_y_train)
            x_valid.extend(tmp_x_valid)
            y_valid.extend(tmp_y_valid)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        # time cropping
        if interval != original_interval:
            x_train = crop_signals(
                x=x_train,
                srate=srate,
                interval_in=original_interval,
                interval_out=interval,
            )
            x_valid = crop_signals(
                x=x_valid,
                srate=srate,
                interval_in=original_interval,
                interval_out=interval,
            )
            x_test = crop_signals(
                x=x_test,
                srate=srate,
                interval_in=original_interval,
                interval_out=interval,
            )

        # channel sampling
        if n_steps_channel_selection is not None:
            x_train = sample_channels(
                x_train,
                data_dict["adjacency_mtx"],
                data_dict["channels"],
                n_steps=n_steps_channel_selection,
                seed_nodes=seed_nodes,
            )
            x_valid = sample_channels(
                x_valid,
                data_dict["adjacency_mtx"],
                data_dict["channels"],
                n_steps=n_steps_channel_selection,
                seed_nodes=seed_nodes,
            )
            x_test = sample_channels(
                x_test,
                data_dict["adjacency_mtx"],
                data_dict["channels"],
                n_steps=n_steps_channel_selection,
                seed_nodes=seed_nodes,
            )

        # swap axes: from (N_examples, C, T) to (N_examples, T, C)
        x_train = np.swapaxes(x_train, -1, -2)
        x_valid = np.swapaxes(x_valid, -1, -2)
        x_test = np.swapaxes(x_test, -1, -2)

        # dataloaders
        train_loader, valid_loader, test_loader = get_dataloader(
            batch_size, (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        )
        datasets = {}
        datasets["train"] = train_loader
        datasets["valid"] = valid_loader
        datasets["test"] = test_loader
        tail_path = os.path.join(
            self.iterator_tag,
            "sub-{0}".format(
                str(dataset.subject_list[target_subject_idx]).zfill(3)
            ),
        )
        return tail_path, datasets
