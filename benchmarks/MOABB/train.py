"""
Training neural networks for MOABB datasets using the new data loading system.

Authors
-------
Victor Cruz, 2025
(Based on original work by Davide Borra and Mirco Ravanelli)
"""

import pickle
import os
import torch
from hyperpyyaml import load_hyperpyyaml

import numpy as np
import logging
import sys
import yaml
import speechbrain as sb
from torch.nn import init
from torch.utils.data import random_split


from dataio.splitters import CrossSessionSplitter, CrossSubjectSplitter


def prepare_dataset(hparams):
    """Create and preprocess dataset using new data loading system."""

    dataset = hparams["EEG_dataset"]
    # 1) Create and update label encoder with all raw labels from the dataset
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.update_from_didataset(dataset, "label")

    # 2) Define a small helper function that calls the encoder
    def encode_label_func(raw_label):
        # This returns a Tensor containing the encoded label
        return label_encoder.encode_label_torch(raw_label)

    # 3) Add a dynamic item that calls our helper function
    dataset.add_dynamic_item(
        encode_label_func, takes=["label"], provides="encoded_label",
    )

    # 4) Change the dataset output keys to produce encoded_label instead of raw "label"
    #    (You can keep "label" too if you want both.)
    dataset.set_output_keys(["encoded_label", "subject", "session", "epoch"])

    return dataset


def prepare_splits(hparams, dataset):
    """Create train/valid/test splits using new splitter system."""

    # Create appropriate splitter
    if hparams["data_iterator_name"] == "leave-one-session-out":
        splitter = CrossSessionSplitter(dataset, leave_k_out=1)
    elif hparams["data_iterator_name"] == "leave-one-subject-out":
        splitter = CrossSubjectSplitter(dataset, leave_k_out=1)
    else:
        raise ValueError(f"Unknown split type: {hparams['data_iterator_name']}")

    # Get specific split based on session index
    split = list(splitter)[hparams["target_session_idx"]]
    train_dataset = split["train"]
    total_len = len(train_dataset)
    val_ratio = hparams["valid_ratio"]
    train_len = int(total_len * (1 - val_ratio))
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(hparams["seed"])
    train_subset, valid_subset = random_split(
        train_dataset, [train_len, val_len], generator=generator
    )
    num_workers = hparams["num_workers"]
    
    if num_workers == None:
        num_workers = torch.get_num_threads() - 1
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=hparams["batch_size"], shuffle=True, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=hparams["batch_size"], num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        split["test"], batch_size=hparams["batch_size"], num_workers=num_workers
    )

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


def load_hparams_and_prepare_data(hparams_file, run_opts, overrides):
    """Load hyperparameters and prepare datasets."""

    # Initial hparams load
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Prepare dataset
    dataset = prepare_dataset(hparams)

    # Update overrides based on actual data shape
    example_batch = next(iter(dataset))

    overrides.update(
        T=example_batch["epoch"].shape[1],  # Time dimension
        C=example_batch["epoch"].shape[0],  # Channel dimension
        n_train_examples=len(dataset),
    )

    # Reload hparams with shape information
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create splits
    datasets = prepare_splits(hparams, dataset)

    # Setup experiment directory
    hparams["exp_dir"] = os.path.join(
        hparams["output_folder"],
        hparams["data_iterator_name"],
        f"sub-{hparams['target_subject_idx']:03d}",
        f"sess-{hparams['target_session_idx']:03d}",
    )

    # Create experiment directory and save config
    sb.create_experiment_directory(
        experiment_directory=hparams["exp_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    return hparams, datasets


def perform_evaluation(brain, hparams, datasets, dataset_key="test"):
    """This function perform the evaluation stage on a dataset and save the performance metrics in a pickle file"""
    brain.log_test_as_valid = dataset_key == "valid"

    min_key, max_key = None, None
    if hparams["test_key"] == "loss":
        min_key = hparams["test_key"]
    else:
        max_key = hparams["test_key"]
    # perform evaluation
    brain.evaluate(
        datasets[dataset_key],
        progressbar=False,
        min_key=min_key,
        max_key=max_key,
    )
    # saving metrics on the desired dataset in a pickle file
    metrics_fpath = os.path.join(
        hparams["exp_dir"], "{0}_metrics.pkl".format(dataset_key)
    )
    with open(metrics_fpath, "wb") as handle:
        pickle.dump(
            brain.last_eval_stats, handle, protocol=pickle.HIGHEST_PROTOCOL
        )


# Keep existing MOABBBrain class and run_experiment function
# Only modify their data handling to work with new dataset format


class MOABBBrain(sb.Brain):
    """Modified Brain class for MOABB experiments with new data format."""

    def init_model(self, model):
        """Initialize neural network modules"""
        for mod in model.modules():
            if hasattr(mod, "weight"):
                if not ("Norm" in mod.__class__.__name__):
                    init.xavier_uniform_(mod.weight, gain=1)
                else:
                    init.constant_(mod.weight, 1)
            if hasattr(mod, "bias"):
                if mod.bias is not None:
                    init.constant_(mod.bias, 0)

    def compute_forward(self, batch, stage):
        """Given an input batch it computes the model output."""
        # Extract EEG data from batch dictionary
        inputs = batch["epoch"].to(self.device)

        # Add channel dimension if needed
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(-1)

        # Perform data augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augment"):
            inputs, _ = self.hparams.augment(
                inputs.squeeze(3),
                lengths=torch.ones(inputs.shape[0], device=self.device),
            )
            inputs = inputs.unsqueeze(3)

        # Normalization
        if hasattr(self.hparams, "normalize"):
            inputs = self.hparams.normalize(inputs)

        return self.modules.model(inputs)

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss given predictions and targets."""
        # Get labels from batch
        targets = batch["encoded_label"].to(self.device)

        # Target augmentation
        N_augments = int(predictions.shape[0] / targets.shape[0])
        targets = torch.cat(N_augments * [targets], dim=0)

        loss = self.hparams.loss(
            predictions,
            targets.squeeze(-1),
            weight=torch.FloatTensor(self.hparams.class_weights).to(
                self.device
            ),
        )

        if stage != sb.Stage.TRAIN:
            # From log to linear predictions
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch["encoded_label"].detach().cpu().numpy())
        else:
            if hasattr(self.hparams, "lr_annealing"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called when a stage (either training, validation, test) starts."""
        if stage != sb.Stage.TRAIN:
            self.preds = []
            self.targets = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a epoch."""
        # Rest of the method remains the same as it handles metrics and checkpointing
        # which don't need to change for the new data format
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            preds = np.array(self.preds)
            y_pred = np.argmax(preds, axis=-1)
            y_true = self.targets
            self.last_eval_stats = {
                "loss": stage_loss,
            }
            for metric_key in self.hparams.metrics.keys():
                self.last_eval_stats[metric_key] = self.hparams.metrics[
                    metric_key
                ](y_true=y_true, y_pred=y_pred)

            # ... rest of the method stays the same ...
            if stage == sb.Stage.VALID:
                # Learning rate scheduler
                if hasattr(self.hparams, "lr_annealing"):
                    old_lr, new_lr = self.hparams.lr_annealing(epoch)
                    sb.nnet.schedulers.update_learning_rate(
                        self.optimizer, new_lr
                    )
                    self.hparams.train_logger.log_stats(
                        stats_meta={"epoch": epoch, "lr": old_lr},
                        train_stats={"loss": self.train_loss},
                        valid_stats=self.last_eval_stats,
                    )
                else:
                    self.hparams.train_logger.log_stats(
                        stats_meta={"epoch": epoch},
                        train_stats={"loss": self.train_loss},
                        valid_stats=self.last_eval_stats,
                    )

                if epoch == 1:
                    self.best_eval_stats = self.last_eval_stats

                # The current model is saved if it is the best or the last
                is_best = self.check_if_best(
                    self.last_eval_stats,
                    self.best_eval_stats,
                    keys=[self.hparams.test_key],
                )
                is_last = (
                    epoch
                    > self.hparams.number_of_epochs - self.hparams.avg_models
                )

                # Check if we have to save the model
                if self.hparams.test_with == "last" and is_last:
                    save_ckpt = True
                elif self.hparams.test_with == "best" and is_best:
                    save_ckpt = True
                else:
                    save_ckpt = False

                # Saving the checkpoint
                if save_ckpt:
                    min_keys, max_keys = [], []
                    if self.hparams.test_key == "loss":
                        min_keys = [self.hparams.test_key]
                    else:
                        max_keys = [self.hparams.test_key]
                    meta = {}
                    for eval_key in self.last_eval_stats.keys():
                        if eval_key != "cm":
                            meta[str(eval_key)] = float(
                                self.last_eval_stats[eval_key]
                            )
                    self.checkpointer.save_and_keep_only(
                        meta=meta,
                        num_to_keep=self.hparams.avg_models,
                        min_keys=min_keys,
                        max_keys=max_keys,
                    )

            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch loaded": self.hparams.epoch_counter.current
                    },
                    test_stats=self.last_eval_stats
                    if not getattr(self, "log_test_as_valid", False)
                    else None,
                    valid_stats=self.last_eval_stats
                    if getattr(self, "log_test_as_valid", False)
                    else None,
                )
                # save the averaged checkpoint at the end of the evaluation stage
                # delete the rest of the intermediate checkpoints
                # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
                if self.hparams.avg_models > 1:
                    min_keys, max_keys = [], []
                    if self.hparams.test_key == "loss":
                        min_keys = [self.hparams.test_key]
                        fake_meta = {self.hparams.test_key: 0.0, "epoch": epoch}
                    else:
                        max_keys = [self.hparams.test_key]
                        fake_meta = {self.hparams.test_key: 1.1, "epoch": epoch}
                    self.checkpointer.save_and_keep_only(
                        meta=fake_meta,
                        min_keys=min_keys,
                        max_keys=max_keys,
                        num_to_keep=1,
                    )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model",
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def check_if_best(
        self, last_eval_stats, best_eval_stats, keys,
    ):
        """Checks if the current model is the best according at least to
        one of the monitored metrics. """
        is_best = False
        for key in keys:
            if key == "loss":
                if last_eval_stats[key] < best_eval_stats[key]:
                    is_best = True
                    best_eval_stats[key] = last_eval_stats[key]
                    break
            else:
                if last_eval_stats[key] > best_eval_stats[key]:
                    is_best = True
                    best_eval_stats[key] = last_eval_stats[key]
                    break
        return is_best


def run_experiment(hparams, run_opts, datasets):
    """Run a single experiment with the new data format."""
    # Calculate class weights
    train_labels = [batch["encoded_label"] for batch in datasets["train"]]
    train_labels = torch.cat(train_labels)
    # train_labels = [ label  for batch in datasets["train"] for label in batch["label"]]
    # unique_labels, label_indices = np.unique(train_labels, return_inverse=True)
    # train_labels_tensor = torch.tensor(label_indices)

    n_examples_perclass = [
        (train_labels == c).sum().item() for c in range(hparams["n_classes"])
    ]
    n_examples_perclass = np.array(n_examples_perclass)
    class_weights = n_examples_perclass.max() / n_examples_perclass
    hparams["class_weights"] = class_weights

    # Setup checkpointer
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join(hparams["exp_dir"], "save"),
        recoverables={
            "model": hparams["model"],
            "counter": hparams["epoch_counter"],
        },
    )

    # Setup logger
    hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
        save_file=os.path.join(hparams["exp_dir"], "train_log.txt")
    )

    # Log dataset info
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory: {hparams['exp_dir']}")

    # Get example batch for logging
    example_batch = next(iter(datasets["train"]))
    logger.info(f"Input shape: {example_batch['epoch'].shape[1:]}")
    logger.info(f"Training set avg value: {example_batch['epoch'].mean():.3f}")

    # Log dataset sizes
    datasets_summary = (
        f"Number of examples: "
        f"{len(datasets['train'].dataset)} (training), "
        f"{len(datasets['valid'].dataset)} (validation), "
        f"{len(datasets['test'].dataset)} (test)"
    )
    logger.info(datasets_summary)

    # Create brain and run training
    brain = MOABBBrain(
        modules={"model": hparams["model"]},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    # if False:  # hparams["dry_run"]:
    #   try:
    #        # Test forward pass with a batch
    #        batch = next(iter(datasets["train"]))
    #        with torch.no_grad():
    #            brain.compute_forward(batch, sb.Stage.TRAIN)
    #        logger.info("✓ Dry run successful - model forward pass works")
    #        raise DryRunComplete("Model validation successful")
    #    except DryRunComplete:
    #        raise
    #    except Exception as e:
    #        logger.error(f"✗ Dry run failed: {str(e)}")
    #        raise

    # Training
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        progressbar=False,
    )

    # Evaluation
    perform_evaluation(brain, hparams, datasets, dataset_key="test")
    brain.hparams.avg_models = 1
    perform_evaluation(brain, hparams, datasets, dataset_key="valid")


if __name__ == "__main__":
    argv = sys.argv[1:]
    # try:
    # loading hparams to prepare the dataset and the data iterators
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    overrides = yaml.load(
        overrides, yaml.SafeLoader
    )  # Convert overrides to a dict
    hparams, datasets = load_hparams_and_prepare_data(
        hparams_file, run_opts, overrides
    )
    # print("Start Training")
    # Run training
    run_experiment(hparams, run_opts, datasets)
    # except DryRunComplete:
    #    print("Dry run successful")
    #    sys.exit(0)
    # except Exception as e:
    #    print(f"Error during execution: {str(e)}")
    #    if overrides.get("dry_run", False):
    #        print("Dry run failed")
    #        sys.exit(1)
    #    raise
