<p align="center">
  <img src="speechbrain-moabb_logo.svg" alt="SpeechBrain-MOABB logo"/>
</p>

# SpeechBrain-MOABB: An open-source Python library for benchmarking deep neural networks applied to EEG signals

This repository provides a set of recipes for processing electroencephalographic (EEG) signals based on the popular [Mother of all BCI Benchmarks (MOABB)](https://github.com/NeuroTechX/moabb), seamlessly integrated with SpeechBrain.

This package facilitates the integration and evaluation of new algorithms (e.g., a novel deep learning architecture or a novel data augmentation strategy) in EEG decoding pipelines based on MOABB-supported tasks, i.e., motor imagery (MI), P300, and steady-state visual evoked potential (SSVEP).
It not only offers an interface for easy model integration and testing but also proposes a fair and robust protocol for comparing different decoding pipelines.

This code can be used for benchmarking new decoding pipelines in a reliable way.
We accompany our code with a benchmark on 9 MOABB datasets (for MI-based, P300-based, and SSVEP-based BCIs) performed using 3 popular deep neural networks for EEG decoding.
Moreover, we also report the main results on the key aspects characterizing the decoding protocol we propose for enabling trustworthy EEG decoding.

For detailed information, please refer to [The link to the official paper will be available soon].

**Important:** This tool requires a Linux environment to run properly.

<!-- ############################################################################################################### -->

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Tutorials](#-tutorials)
- [Installation](#-installation)
- [Datasets and Recipes](#-datasets-and-recipes)
  - [Training Strategies](#training-strategies)
- [Quickstart](#-quickstart)
  - [Training and Evaluation for a Specific Subject and Session](#training-and-evaluation-for-a-specific-subject-and-session)
  - [Run a Complete Experiment on a Given Dataset](#run-a-complete-experiment-on-a-given-dataset)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Overview](#overview)
    - [Incorporating Orion Flags in Hparam Files](#incorporating-orion-flags-in-hparam-files)
    - [Multi-Step Hyperparameter Optimization](#multi-step-hyperparameter-optimization)
    - [Workflow of the Script](#workflow-of-the-script)
    - [Running Hyperparameter Optimization](#running-hyperparameter-optimization)
    - [Output Structure](#output-structure)
- [‍Incorporating Your Model](#-incorporating-your-model)
- [Results](#-results)
- [Validation of the proposed decoding protocol](#-validation-of-the-proposed-decoding-protocol)
  - [Hyperparameter search on all participants or on a subset of participants](#hyperparameter-search-on-all-participants-or-on-a-subset-of-participants)
  - [Two-step vs. one-step hyperparameter search](#two-step-vs-one-step-hyperparameter-search)
  - [Sequential model-based search vs. random search](#sequential-model-based-search-vs-random-search)
  - [Performance variability due to random initialization](#performance-variability-due-to-random-initialization)
- [Contact](#-contact)
- [Citing](#-citing)
<!-- ############################################################################################################### -->
# 🎓 Tutorials
We provide tutorials for introducing users to SpeechBrain-MOABB, covering its main aspects.
Users can open and run tutorials directly in Google Colaboratory in a straightforward way.

Specifically, we provide:


* Tutorial no. 1 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x3kNX65FelDhyl6ooNyFK8pVXsLxTm_N)
 **[Setting up EEG decoding](https://drive.google.com/file/d/1x3kNX65FelDhyl6ooNyFK8pVXsLxTm_N/view?usp=sharing)**.
In this tutorial, we provide a brief introduction to SpeechBrain-MOABB and its first command-line interface.
This interface allows users to run a full experiment on a target decoding problem (e.g., P300 decoding on a MOABB dataset) with known hyperparameters (i.e., no hyperparameter tuning is performed).
Models are trained and evaluated adopting a multi-seed initialization strategy, for providing performance estimates robust to fluctuations due to random initialization.
* Tutorial no. 2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w-VTkYn4MPKa1PZceKZ7MkBDMVi1zi4J)
 **[Setting up hyper-parameter tuning](https://drive.google.com/file/d/1w-VTkYn4MPKa1PZceKZ7MkBDMVi1zi4J/view?usp=sharing)**.
This tutorial covers the second command-line interface of SpeechBrain-MOABB, devoted to performing multi-step hyperparameter tuning of the hyperparameters characterizing the entire decoding pipeline.
Notably, the tuned hyperparameters can include data pre-processing, network architecture, network training, and data augmentation steps of the deep learning-based EEG decoding pipeline.
As for the previous command-line interface, here models are trained and evaluated by adopting a multi-seed initialization strategy, for providing robust performance estimates.
* Tutorial no. 3 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W_0YRAqioceKo3XPdLB9A2d7yT4iOO50)
 **[Integrating braindecode models](https://drive.google.com/file/d/1W_0YRAqioceKo3XPdLB9A2d7yT4iOO50/view?usp=sharing)**.
This tutorial shows how to use braindecode models in SpeechBrain-MOABB, designing a robust and comprehensive decoding pipeline with SpeechBrain-MOABB while taking advantage of the large set of models supported in braindecode.

Moreover, we also provide a [tutorial](https://drive.google.com/file/d/1EmgYqHv7xdhUpkrNL5eVMwF054__MGWq/view?usp=sharing)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EmgYqHv7xdhUpkrNL5eVMwF054__MGWq) for replicating the results reported in the paper associated with SpeechBrain-MOABB, to ensure transparency of the results reported in the paper.
<!-- ############################################################################################################### -->
# 🛠 Installation

To set up SpeechBrain-MOABB, follow these steps:


1. Clone the benchmark repository and install SpeechBrain:
   ```shell
   git clone https://github.com/speechbrain/benchmarks.git
   cd benchmarks
   git submodule update --init --recursive
   cd speechbrain
   pip install -r requirements.txt
   pip install -e .
   ```

2. Navigate to `<path-to-repository>/benchmarks/MOABB` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements.

**⚠️ Note**: We tested the toolkit with Python 3.10. While it also works with Python 3.11, some users have reported issues, particularly related to GPy. You can find more information about this in [issue](https://github.com/speechbrain/benchmarks/issues/21).

**Notes on MOABB and MNE:**

The benchmark results presented here were generated using MOABB version 0.4.6.
The code, however, remains compatible with newer MOABB versions.
It is important to be aware that starting from MOABB version 1.0, there have been changes to the naming conventions for sessions.
As an example, in BNCI2014001, the session labeled as `session_T` in previous versions is now referred to as `0train`, and `session_E` is now named `1test` in MOABB versions 1.0 and above.


The code relies on [MNE](https://mne.tools/stable/index.html), which, by default, stores a config file at `$HOME/.mne/mne-python.json` and downloads data to `$HOME/mne-data`.
However, in some cases, the home directory may not exist, have storage limitations, or be on a shared filesystem where data operations are restricted by the system admin.

If you need to set up a different folder for MNE, follow these steps:

1. Open your bash shell and execute the following command to set the environment variable `_MNE_FAKE_HOME_DIR` to your preferred folder:

   ```bash
   export _MNE_FAKE_HOME_DIR='your/folder'
   ```

2. Launch a Python session and import the MNE module. This action will create a new configuration file at `your/folder/.mne/mne-python.json`.

3. Open the newly created file (`your/folder/.mne/mne-python.json`) and set the `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` variables to the folders you want to use for MNE data and MOABB datasets, respectively.

By following these steps, you can ensure that MNE uses the specified folder for configuration and data storage.
<!-- ############################################################################################################### -->
# ⚡ Datasets and Recipes

This package leverages datasets supported by [MOABB](https://neurotechx.github.io/moabb/datasets.html).
Specifically, it comes with recipes for the following [datasets](http://moabb.neurotechx.com/docs/datasets.html):


| Dataset ID | Task | nsbj | nsess |
|------------|-------------|-----|-----|
|BNCI2014001 | Motor Imagery | 9 | 2 |
|BNCI2014004 | Motor Imagery | 9 | 5 |
|BNCI2015001 | Motor Imagery | 12 | 2 |
|Lee2019_MI | Motor Imagery | 54 | 2 |
|Zhou201 | Motor Imagery | 4 | 3 |
|BNCI2014009 | P300 | 10 | 3 |
|EPFLP300 | P300 | 8 | 4 |
|bi2015a | P300 | 43 | 3 |
|Lee2019_SSVEP | SSVEP | 54 | 2 |

The EEG datasets are automatically downloaded when running the provided recipes.
Furthermore, the code is designed to allow easy integration of any other dataset from MOABB, as well as the ability to plug and test various deep neural networks.
The benchmark includes recipes for using the datasets mentioned above with popular models such as:
- [EEGNet](https://arxiv.org/pdf/1611.08024.pdf)
- [ShallowConvNet](https://arxiv.org/pdf/1703.05051.pdf)
- [EEGConformer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9991178)

Users can easily integrate their own PyTorch models into our benchmark by following the instructions provided in the "Incorporating Your Model" section below.


## Training Strategies
EEG recordings involve recording the brain activity from a subject using multiple EEG sensors placed on their head, resulting in a multi-channel signal (one for each sensor).

These recordings can be performed while the subject is engaged in specific tasks, such as *motor imagery*, where they are asked to think about a particular movement.

Multiple recordings, each involving the same subject undertaking the same task, are typically conducted. These recordings are referred to as *sessions*.

One of the distinctive features of EEG tasks compared to other popular machine learning tasks, such as speech processing or computer vision, is the relatively low amount of data available for each subject. Additionally, due to the cost of recording brain activity, the number of subjects is not large.

Normally, two common strategies are used during the training phase: *Leave-One-Session-Out* and *Leave-One-Subject-Out* cross-validation.
Both cross-validation approaches are supported in SpeechBrain-MOABB.

* **Leave-One-Session-Out**:
  For each subject, we reserve one session as a test set and use the remaining sessions for training neural networks.
  We thus train different neural networks, each excluding a different session.
  We repeat this process for all subjects and then average the performance to asses the final performance of our models.

* **Leave-One-Subject-Out**:
  In this challenging condition, we reserve one subject as the test set while training using the data from all the other subjects.
  This approach is challenging because each subject has a unique brain activity pattern, making it difficult to successfully leverage data from other subjects.
<!-- ############################################################################################################### -->

# 🚀 Quickstart

**Note:** Before proceeding with the experiments, make sure that you have installed the additional dependencies listed in the `extra_requirements.txt` file.
Please, read the content above as well.

## Training and Evaluation for a Specific Subject and Session

Let's now dive into how to train a model using data from a single subject and session. Follow the steps below to run this experiment:

```bash
python train.py hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data --output_folder=results/MotorImagery/BNCI2014001/ --target_subject_idx=0 --target_session_idx=1 --data_iterator_name=leave-one-session-out
```

**Note**: To run the model on the CPU, simply include --device='cpu' in the training command.

In this example, we will train EEGNet for Motor Imagery using the BNCI2014001 dataset.
Specifically, we will train the model using data from *subject 0*.
The data recorded in *session 1* of *subject 0* will be used for testing, while all the other sessions will be used for training.

The data will be automatically downloaded to the specified `data_folder`, and a cached version of the data will be stored in `cached_data_folder` for future reuse.

Ensure that your local machine has internet access. If it does not, you can download the data in advance and store it in the specified data_folder.

The results, including training logs and checkpoints, will be available in the output folder specified in the hyperparameter (hparam) file.

In the example above, the output folder contains the models trained for sessions 'T' and 'E'. Within each subfolder, you can find a variety of files generated during model training.

For instance, the train_log.txt file appears as follows:

```
epoch: 1, lr: 1.95e-05 - train loss: 1.39 - valid loss: 1.39, valid f1: 1.84e-01, valid acc: 2.14e-01, valid cm: [[4 0 6 4]
 [6 0 6 2]
 [3 0 5 6]
 [4 1 6 3]]
epoch: 2, lr: 4.03e-05 - train loss: 1.33 - valid loss: 1.39, valid f1: 2.51e-01, valid acc: 3.04e-01, valid cm: [[ 1  1 11  1]
 [ 0  4 10  0]
 [ 1  1 11  1]
 [ 0  1 12  1]]
epoch: 3, lr: 6.11e-05 - train loss: 1.25 - valid loss: 1.39, valid f1: 3.13e-01, valid acc: 3.57e-01, valid cm: [[5 7 2 0]
 [4 9 1 0]
 [4 4 6 0]
 [9 4 1 0]]

...

epoch loaded: 862 - test loss: 5.94e-02, test f1: 8.57e-01, test acc: 8.57e-01, test cm: [[13  1  0  0]
 [ 3 11  0  0]
 [ 0  0 12  2]
 [ 0  0  2 12]]

```

This log file reports various training metrics for each epoch, including train/validation losses, accuracies, and a confusion matrix that provides insights into misclassified classes.

Additionally, you can find detailed performance metrics for both validation and testing in files named `valid_metrics.pkl` and `test_metrics.pkl`."


## Run a Complete Experiment on a Given Dataset

To train models using either the *Leave-One-Subject-Out* or *Leave-One-Session-Out* approach and then average their performance, we have developed a convenient bash script called `run_experiment.sh`.
This script orchestrates the necessary loops for easy execution and represents the first command-line interface of SpeechBrain-MOABB.

To run a full training experiment, use the following command:

```bash
./run_experiments.sh --hparams hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder eeg_data --output_folder results/MotorImagery/BNCI2014001/EEGNet --nsbj 9 --nsess 2 --nruns 10 --train_mode leave-one-session-out --device=cuda
```

For further details on arguments and customization options, consult `./run_experiments.sh`.

This command will execute the `leave-one-session-out` training on the BNCI2014001 dataset for motor imagery using the EEGNet.yaml configuration.
The script will loop over 9 subjects and 2 sessions, running the experiment 10 times (--nruns 10) with different initialization seeds to ensure robustness.
Running multiple experiments with varied seeds and averaging their performance is a recommended practice to improve result significance.

Please note that due to the thorough evaluation process we employ, running an experiment may require some time. In particular, it may take approximately 8 hours when executed on an NVIDIA V100 GPU.
The results of each experiment are saved in the specified output folder. To view the final aggregated performance, refer to the `aggregated_performance.txt` file.
The `aggregated_performance.txt` file should look like this:

```
---- leave-one-session-out ----

session_E acc [0.728400 0.722600 0.727200 0.735000 0.727200 0.728400 0.733400 0.724500 0.734600 0.726900] avg: 0.728820 ± 0.003990
session_T acc [0.727600 0.736500 0.729900 0.736900 0.737300 0.736500 0.748500 0.734200 0.739200 0.737700] avg: 0.736430 ± 0.005306


Aggregated results

acc [0.728009 0.729552 0.728588 0.735918 0.732253 0.732446 0.740934 0.729360 0.736883 0.732253] avg: 0.732620 ± 0.003950
[{'name': 'objective', 'type': 'objective', 'value': 0.26738040123456785}]

```

The system's performance should closely match the values reported in the first row of the table in [Results](#-results) section. Specifically, please refer to the line corresponding to `/MotorImagery/BNCI2014001/EEGNet.yaml`.

Please be aware that there may be slight variability each time the experiment is run, primarily due to different seeds being used in each run. For example, you may observe accuracy values like `0.732620` in the current run, while the original table reports `0.731559`. We conduct 10 runs and average the results to minimize this variability and provide a more accurate performance estimate.

In addition to the `aggregated_performance.txt` file, the output folder includes further information:

- Each run's performance details are reported in individual files, such as `run1_results.txt`.
- For each run, a dedicated folder (e.g., `run1`) contains the models trained for each subject, complete with logging information.

Feel free to explore the output folder and become familiar with the provided output.

By default, the hyperparameters in the yaml files are those determined during hyperparameter tuning (as shown below).


**Default Values:**
- By default, the training modality is set to `leave-one-session-out`. If you prefer to use `leave-one-subject-out`, simply add the flag `--train_mode=leave-one-subject-out`.
- The default evaluation metric is accuracy (acc). If you wish to use F1 score instead, use the flag `--eval_metric=f1`.
- By default, the evaluation is conducted on the test set. To use the dev set instead, use the flag `--eval_set=dev`.
- Without specifying the `--seed flag`, a random seed is used.
- Beyond the flags expected by the `./run_experiments.sh` script, you can use additional flags to override any value declared in the hparam file. In the example above, we changed the number of epochs to 2.

**Note**: This script operates under the assumption that you are utilizing a Linux-based system. In this scenario, we offer a bash script instead of a Python script due to its inherent suitability for effectively orchestrating multiple training loops across various subjects and sessions.

**Important:** The number of subjects (`--nsbj`) and sessions (`--nsess`) is dataset-dependent. Refer to the dataset [dataset table above](#-datasets-and-recipes) for these details. When executing a training experiment on a different dataset or model, please modify both the hparam file and adjust the subject and session counts accordingly.

## Hyperparameter Tuning

Efficient hyperparameter tuning is paramount when introducing novel models or experimenting with diverse datasets.
Our benchmark establishes a standardized protocol for hyperparameter tuning, utilizing [Orion](https://orion.readthedocs.io/en/stable/) to ensure fair model comparisons.

The standardized protocol we propose is based on multi-step hyperparameter search, for addressing the search in a large hyperparameter space, and on multi-seed initialization, for providing robust performance estimates.
Note that several aspects affecting this protocol underwent deep investigation.
These include multi-step vs. single-step search, the search algorithm used, the number of participants for hyperparameter search, and the number of random seeds for providing a stable decoding performance.
Refer to [protocol results below](#-validation-of-the-proposed-decoding-protocol) for the results on the key aspects characterizing the decoding protocol.
### **Overview**

Hyperparameter tuning is orchestrated through the `./run_hparam_optimization.sh` script, which oversees the execution of multiple hyperparameter trials via `run_experiments.sh`.
This script represents the second command-line interface of SpeechBrain-MOABB.

Please keep in mind the following points:
- In certain scenarios, you may find it advantageous to retain separate experiment folders for each hyperparameter trial. You can achieve this by using the `--store_all True` flag. Conversely, setting it to false will condense results within a singular folder, a space-saving measure.
- The script effectively manages all essential phases for executing multi-step hyperparameter tuning. It further assesses the final performance on the test set using the optimal hyperparameters, with performance being averaged across `--nruns_eval` iterations to enhance result significance.

### **Incorporating Orion Flags in Hparam Files**

The script assumes that Orion flags are directly included in the specified YAML hparam file using comments. To optimize, for instance, the dropout parameter within a defined range, you need to have the following line in the YAML file:

```yaml
dropout: 0.1748  # @orion_step1: --dropout~"uniform(0.0, 0.5)"
```

### **Multi-Step Hyperparameter Optimization**

Our method supports multi-step hyperparameter optimization.

In practice, we can optimize a subset of hyperparameters while keeping the others fixed.

After finding their optimal values, we utilize them as a foundation for optimizing another set of hyperparameters.

This approach has consistently demonstrated superior results in our benchmark, especially when distinguishing between training and architectural hyperparameters versus data augmentation hyperparameters.

We thus propose a two-phase hyperparameter optimization strategy: in phase 1, we optimize hyperparameters related to the neural network architecture, while in phase 2, our focus shifts exclusively to data augmentation hyperparameters.


To optimize a hyperparameter in a second step, follow this syntax in the YAML file:

```yaml
snr_white_low: 9.1 # @orion_step2: --snr_white_low~"uniform(0.0, 15, precision=2)"
```

Users have the flexibility to define multiple optimization steps based on their experimental protocol, although two steps, as recommended, often suffice.

### **Workflow of the Script**

The script operates as follows:

1. Scans the specified hparam file for Orion flags.
2. Executes hyperparameter tuning using the `orion-hunt` command.
3. Captures and saves the best hyperparameters for reference via `torch-info`.
4. Continues until flags like `@orion_step<stepid>` are encountered in the YAML file.

### **Running Hyperparameter Optimization**

You can conduct hyperparameter optimization with commands similar to the following:

```bash
./run_hparam_optimization.sh --exp_name 'EEGNet_BNCI2014001_hopt' \
                             --output_folder results/MotorImagery/BNCI2014001/EEGNet/hopt \
                             --data_folder eeg_data/ \
                             --hparams hparams/MotorImagery/BNCI2014001/EEGNet.yaml \
                             --nsbj 9 --nsess 2 \
                             --nsbj_hpsearch 9 --nsess_hpsearch 2 \
                             --nruns 1 \
                             --nruns_eval 10 \
                             --eval_metric acc \
                             --train_mode leave-one-session-out \
                             --exp_max_trials 50
```

For further details on arguments and customization options, consult `./run_hparam_optimization.sh`.

Note that hyperparameter tuning may take several hours (up to several days) depending on the model complexity and dataset.
To speed up hyper-parameter tuning you can consider reducing the number of subjects and sessions used during hyper-parameter tuning, by setting the `--nsbj_hpsearch ` and `--nsess_hpsearch` flags.
As an example, in the previous command you can set `--nsbj_hpsearch 3 --nsess_hpsearch 2` to run hyper-parameter tuning only on a subset of subjects / sessions.
Of course, the final evaluation will be performed on the entire dataset (on all subjects and sessions).

Our protocol ensures a model comparison that is as fair as possible.
All reported results reported below are achieved with the same hyperparameter tuning methodology, enabling fair assessments across diverse models.

As evident from the example, you need to configure the hyperparameter file, specify the number of subjects (nsbj), and set the number of sessions (nsess).

The [table above](#-datasets-and-recipes) provides these values for each compatible dataset.

When it comes to training the model utilizing the leave-one-subject-out approach, simply employ the `--train_mode leave-one-subject-out` flag.

**Notes:**
- By default, model training is performed on GPU. However, in case you do not have any GPU available on your machine, you can train models on CPU by specifying the `--device cpu` flag.
- To monitor the status of the hyperparameter optimization, simply enter the following command: `orion status --all`. Ensure that you have added the necessary variables required by orion to your bash environment. You can achieve this by executing the following code within your terminal:

```bash
export ORION_DB_ADDRESS=results/MotorImagery/BNCI2014001/EEGNet/hopt/EEGNet_BNCI2014001_hopt.pkl
export ORION_DB_TYPE=pickleddb
```

Please note that the value of the `ORION_DB_ADDRESS` variable will vary depending on the experiment. Adjust it accordingly.

- If needed, you can interrupt the code at any point, and it will resume from the last completed trial.

- The quantities of subjects (`--nsbj`, `--nsbj_hpsearch`) and of sessions (`--nsess`, `--nsess_hpsearch`) are dataset-dependent. Please consult the [table above](#-datasets-and-recipes) for this information.
 When conducting a hyperparameter optimization experiment using an alternative dataset or model, kindly adjust both the hparam file and the subject/session counts accordingly.

- If you intend to perform multiple repetitions of the same hparam optimization, it is necessary to modify the `--exp_name`.

- This script is designed for a Linux-based system. In this context, we provide a bash script instead of a Python script due to its natural ability to orchestrate diverse training loops across various subjects and sessions.


### **Output Structure**

Results are saved within the specified output folder (`--output_folder`).
In the following, we report a typical output organization obtained during leave-one-session-out training, using multi-step hyperparameter search with 2 steps and multi-seed training and evaluation with 10 seeds.

```
output folder
│   best_hparams.yaml # YAML file containing optimal hyperparameters after all hyperparameter search steps (copy of the ones contained in 'best' folder)
│   hyperparameter-search.pkl # result history of hyperparameter search
└───best # multi-seed final training and evaluation
│   └───sjsHfg # random name
│       │   flags.txt # options used while launching SpeechBrain-MOABB command-line interface
│       │   aggregated_performance.txt # aggregated performance evaluation across seeds
│       │   run1_results.txt # performance evaluation for one seed
│       │   run2_results.txt
│       │   ...
│       │   run10_results.txt
│       └───run1 # one run of multi-seed random training and evaluation
│       │   └───1986  # seed used
│       │       └───leave-one-session-out  # training strategy
│       │           └───sub-001 # subject
│       │           │   └───ses-01 # held-out session (used as test set)
│       │           │   │   └───save # saved model checkpoints
│       │           │   │   │   env.log # environment details (Python, Python libraries, etc.)
│       │           │   │   │   log.txt # complete log file
│       │           │   │   │   train_log.txt # training log file
│       │           │   │   │   model.txt # model details
│       │           │   │   │   train.py # Python script used for training and evaluation
│       │           │   │   │   hyperparameters.yaml # hyperparameter YAML file
│       │           │   │   │   valid_metrics.pkl # evaluation metrics on validation set
│       │           │   │   │   test_metrics.pkl # evaluation metrics on test set
│       │           │   └───... # other held-out sessions
│       │           └───... # other subjects
│       └───run2
│       │   └───...
│       └───...
│       └───run10
│           └───...
└───step1 # step of multi-step hyperparameter search
│   │   best_hparams.txt # optimal hyperparameters found during the search
│   │   best_hparams.yaml # YAML file containing all hyperparameters, with the tuned hyperparameters at their optimal value
│   │   exp.tar.gz # compressed directory with all results obtained during the search
│   │   orion-info.txt # results from Orion
│   │   orion_hunt_command.txt # command used for running Orion
└───step2
│   │   ...
```

Notes:
- Random names are widely used for directories containing results from the performed experiments (e.g., within 'best' folder)
- To circumvent the generation of excessive files and folders within the directory containing hyperparameter search results (e.g., 'step1' folder), which can be an issue on certain HPC clusters due to file quantity restrictions, consider activating the `--compress_exp True` option.
- Note that, the organization of the 'best' folder obtained by running the command-line interface `run_hparam_optimization.sh` corresponds to the organization of the output folder obtained with the previous command-line interface (i.e., `run_experiments.sh`)
<!-- ############################################################################################################### -->
# 📝 ‍Incorporating Your Model

Let's now assume you've designed a neural network in PyTorch and wish to integrate it into our benchmark.
You're in luck because we've made this step as simple as possible for you!
Here are the steps you should follow:

1. Write your model's code in a Python library saved in `benchmarks/MOABB/models` (e.g., `benchmarks/MOABB/models/my_model.py`).
Ensure that your model is compatible with the EEG task, considering varying input channels and variable-length inputs across different datasets.

2. Create a YAML file for each dataset you want to experiment with. Thankfully, you don't have to start from scratch. For example, if you're working with BNCI2014001 (Motor Imagery/), copy `benchmarks/MOABB/hparams/MotorImagery/BNCI2014001/EEGNet.yaml` and save it in the same folder with a different name (e.g., `my_model.yaml`).

3. Edit the relevant section of your `my_model.yaml`. Redefine the `model:` to reference your custom model (e.g., `model: !new:models.my_model.my_model`).

4. Ensure you include the hyperparameters specific to your model, along with the ranges you'd like to explore during hyperparameter tuning.

5. Now, follow the instructions above to run an experiment and perform the necessary hyperparameter tuning.

**Note**: If you're not familiar with YAML, you can refer to our [HyperPyYAML tutorial](https://speechbrain.github.io/tutorial_basics.html) on the SpeechBrain website for guidance.
<!-- ############################################################################################################### -->

# 📈 Results
Here, we report some results while benchmarking three popular EEG deep learning-based models for decoding motor imagery, P300, and SSVEP with SpeechBrain-MOABB.

Performance metrics were computed on each held-out session (stored in the metrics.pkl file) and reported here averaged across sessions and subjects, displaying the average value ± standard deviation across 10 random seeds.

| Release | Task | Hyperparams file | Training strategy | Key loaded model | Performance (test set) |  GPUs |
|:-------------:|:-------------:|:---------------------------:|:---------------------------:|  -----:| -----:| :-----------:|
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.731559±0.003888 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014001/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.695795±0.003748 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014001/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.675810±0.006926 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014004/EEGNet.yaml | leave-one-session-out |  'acc'| 0.812062±0.001888 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2014004/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.784813±0.003038 | 1xNVIDIA V100 (16 GB) |
| 27-10-02 | Motor imagery | /MotorImagery/BNCI2014004/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.799148 ± 0.002082  | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.810646±0.002113 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015001/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.828646±0.003781 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/BNCI2015001/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.751688±0.009589 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Lee2019_MI/EEGNet.yaml | leave-one-session-out |  'acc'| 0.694278±0.003121 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Lee2019_MI/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.657500±0.004488 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Lee2019_MI/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.651333±0.008495 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Zhou2016/EEGNet.yaml | leave-one-session-out |  'acc'| 0.843619±0.005637 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Zhou2016/ShallowConvNet.yaml | leave-one-session-out |  'acc'| 0.826854±0.005277 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | Motor imagery | /MotorImagery/Zhou2016/EEGConformer.yaml | leave-one-session-out |  'acc'| 0.839601±0.005769 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | P300 | /P300/EPFLP300/EEGNet.yaml | leave-one-session-out |  'f1'| 0.634613±0.003840 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | P300 | /P300/BNCI2014009/EEGNet.yaml | leave-one-session-out |  'f1'| 0.754958±0.001643 | 1xNVIDIA V100 (16 GB) |
| 27-10-02 | P300 | /P300/bi2015a/EEGNet.yaml | leave-one-session-out |  'f1'| 0.723952±0.002445 | 1xNVIDIA V100 (16 GB) |
| 23-10-02 | SSVEP | /SSVEP/Lee2019_SSVEP/EEGNet.yaml | leave-one-session-out |  'acc'| 0.916148±0.002436 | 1xNVIDIA V100 (16 GB) |

Notes:
- To ensure transparency and reproducibility, we release the output folder containing model checkpoints and training logs. You can access **checkpoints** for each model and dataset, complete with hyperparameter tuning [here](https://www.dropbox.com/sh/ux0i0suljojonmb/AABsTBpEKCTmVE784yQw-WGMa?dl=0).
- The experiments can be conducted on any GPU with a memory capacity of 12 GB or higher.
- ShallowConvNet and EECConformer models are excluded for P300 and SSVEP experiments, as these models are tailored for Motor Imagery tasks.

<!-- ############################################################################################################### -->
# 📈 Validation of the proposed decoding protocol
In the following, we report the main results that were obtained by changing the key aspects of the decoding protocol, such as:
* The number of participants used during hyperparameter search
* Multi-step hyperparameter search (2-step search vs. 1-step search)
* The hyperparameter search algorithm (sequential model-based search vs. random search)
* The performance fluctuations due to random initialization of neural networks

The performance was computed on each held-out session (stored in the metrics.pkl file) and was averaged across sessions and subjects.
## Hyperparameter search on all participants or on a subset of participants
Hyperparameter search was performed using all the participants available or a subset of participants, for reducing computational time.

In a first case of study, we use a subset of formed by 3 or 5 participants (5 participants for the largest datasets among the considered ones, i.e., Lee2019_MI, Lee2019_SSVEP).
From our results, using a subset of participants slightly worsened the performance only for 4 out of 9 datasets from 0.73 to 5.3%, while at the same time reducing the computational time required (from 321.8 to 55.9 hours, on average across datasets).

We also report results using a subset formed by only 1 participant, as a last case of study. In this case, the performance worsened up to 14.6%.

The table presented below illustrates the performance difference observed when utilizing the entire set of subjects compared to using only a subset of them (a negative number indicates a decline in performance).

| Task | Hyperparams file | Training strategy | Key loaded model | Mean performance (test set): all - subset of 3-5 participants | Mean performance (test set): all - subset of 1 participant|  GPUs |
|:-------------:|:---------------------------:|:---------------------------:|  -----:| -----:|-----:| :-----------:|
| Motor imagery | /MotorImagery/BNCI2014001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0318  | 0.0176 |1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2014004/EEGNet.yaml | leave-one-session-out |  'acc'| -0.0076| -0.0008 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2015001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0350 | 0.0208 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Lee2019_MI/EEGNet.yaml | leave-one-session-out |  'acc'| -0.0532 | -0.1088 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Zhou2016/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0091 | -0.0504 | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/EPFLP300/EEGNet.yaml | leave-one-session-out |  'f1'| -0.0140 | -0.0423 |1xNVIDIA V100 (16 GB) |
| P300 | /P300/BNCI2014009/EEGNet.yaml | leave-one-session-out |  'f1'| -0.0073 | -0.0206 |1xNVIDIA V100 (16 GB) |
| P300 | /P300/bi2015a/EEGNet.yaml | leave-one-session-out |  'f1'| 0.0101 | -0.0117 | 1xNVIDIA V100 (16 GB) |
| SSVEP | /SSVEP/Lee2019_SSVEP/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0144 |-0.1456 |1xNVIDIA V100 (16 GB) |


## Two-step vs. one-step hyperparameter search
Hyperparameter search was performed on the entire search space in a single step (1-step search) or on subspaces of the entire search space performing two sequential spaces (2-step search).
From our results, two-step search was superior to single-step search for 6 out of 9 datasets used, with improvements up to 10.9%.

The table presented below illustrates the performance difference observed when utilizing 1-step search vs. 2-step search (a negative number indicates a decline in performance).

| Task | Hyperparams file | Training strategy | Key loaded model | Mean performance (test set): 2-step - 1-step search |  GPUs |
|:-------------:|:---------------------------:|:---------------------------:|  -----:| -----:| :-----------:|
| Motor imagery | /MotorImagery/BNCI2014001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0324  |1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2014004/EEGNet.yaml | leave-one-session-out |  'acc'| -0.0035 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2015001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0465 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Lee2019_MI/EEGNet.yaml | leave-one-session-out |  'acc'| -0.0207 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Zhou2016/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0305 | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/EPFLP300/EEGNet.yaml | leave-one-session-out |  'f1'| -0.0109 | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/BNCI2014009/EEGNet.yaml | leave-one-session-out |  'f1'| 0.0136 |1xNVIDIA V100 (16 GB) |
| P300 | /P300/bi2015a/EEGNet.yaml | leave-one-session-out |  'f1'| 0.0127 |  1xNVIDIA V100 (16 GB) |
| SSVEP | /SSVEP/Lee2019_SSVEP/EEGNet.yaml | leave-one-session-out |  'acc'| 0.1088 | 1xNVIDIA V100 (16 GB) |

## Sequential model-based search vs. random search
Hyperparameter search was performed using TPE (configuration file at: `hparams/orion/hparams_tpe.yaml`) or using random search (configuration file at: `hparams/orion/hparams_random_search.yaml`).
From our results, sequential model-based search (TPE-based) was superior to random search for 7 out of 9 datasets used, with improvements up to 5.7%.

The table presented below illustrates the performance difference observed when utilizing TPE search vs. random search (a negative number indicates a decline in performance).

| Task | Hyperparams file | Training strategy | Key loaded model | Mean performance (test set): TPE - random search |  GPUs |
|:-------------:|:---------------------------:|:---------------------------:|  -----:| -----:| :-----------:|
| Motor imagery | /MotorImagery/BNCI2014001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0567  |1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2014004/EEGNet.yaml | leave-one-session-out |  'acc'| -0.0063 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2015001/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0535 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Lee2019_MI/EEGNet.yaml | leave-one-session-out |  'acc'| -0.0105 | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Zhou2016/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0535 | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/EPFLP300/EEGNet.yaml | leave-one-session-out |  'f1'| 0.0036 | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/BNCI2014009/EEGNet.yaml | leave-one-session-out |  'f1'| 0.0137 |1xNVIDIA V100 (16 GB) |
| P300 | /P300/bi2015a/EEGNet.yaml | leave-one-session-out |  'f1'| 0.0142 |  1xNVIDIA V100 (16 GB) |
| SSVEP | /SSVEP/Lee2019_SSVEP/EEGNet.yaml | leave-one-session-out |  'acc'| 0.0224 | 1xNVIDIA V100 (16 GB) |

## Performance variability due to random initialization
After hyperparameter search, the final models were trained and evaluated with 100 random seeds and the standard deviation was computed across averages across 1 or 10 seeds.
From our results, using 10 random seeds the performance variability was less than 1% for all datasets (9 out of 9 datasets), while with only 1 random seed the performance variability was up to 4.9639%.

The table presented below illustrates the performance variability (std. dev.) resulting from using averages across 1 random seed or 10 random seeds (i.e., no multi-seed random initialization vs. 10-seed random initialization).

| Task | Hyperparams file | Training strategy | Key loaded model | std. dev.: 1 seed (%)| std. dev.: 10 seeds (%)|  GPUs |
|:-------------:|:---------------------------:|:---------------------------:|  -----:| -----:|-----:| :-----------:|
| Motor imagery | /MotorImagery/BNCI2014001/EEGNet.yaml | leave-one-session-out |  'acc'| 1.3470  |0.4251  |1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2014004/EEGNet.yaml | leave-one-session-out |  'acc'| 0.9280 |0.2016  | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/BNCI2015001/EEGNet.yaml | leave-one-session-out |  'acc'| 1.5500 |0.5143  | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Lee2019_MI/EEGNet.yaml | leave-one-session-out |  'acc'| 4.9639 |0.8843  | 1xNVIDIA V100 (16 GB) |
| Motor imagery | /MotorImagery/Zhou2016/EEGNet.yaml | leave-one-session-out |  'acc'| 1.6872 |0.4132  | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/EPFLP300/EEGNet.yaml | leave-one-session-out |  'f1'| 1.8622 |0.5874  | 1xNVIDIA V100 (16 GB) |
| P300 | /P300/BNCI2014009/EEGNet.yaml | leave-one-session-out |  'f1'| 1.6558 |0.4564  |1xNVIDIA V100 (16 GB) |
| P300 | /P300/bi2015a/EEGNet.yaml | leave-one-session-out |  'f1'| 0.9080 | 0.1607  | 1xNVIDIA V100 (16 GB) |
| SSVEP | /SSVEP/Lee2019_SSVEP/EEGNet.yaml | leave-one-session-out |  'acc'| 1.6279 |0.7560  | 1xNVIDIA V100 (16 GB) |
<!-- ############################################################################################################### -->
# 📧 Contact

For any questions or inquiries, feel free to reach Davide Borra, Ph.D., University of Bologna ([davide.borra2@unibo.it](mailto:davide.borra2@unibo.it)).
<!-- ############################################################################################################### -->
# 📖 Citing

If you use SpeechBrain-MOABB for your research or business, please cite:

[The link to the official paper will be available soon]

Please also cite SpeechBrain:

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
