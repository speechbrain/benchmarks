# MUSDB Separation Task

This folder defines the **MUSDB source separation benchmark** within DASB (Discrete Audio Separation Benchmark). It enables evaluating discrete audio representations on **music source separation**, using the [MUSDB-18 dataset](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav).

## Overview

The goal of this task is to perform **source separation** from musical mixtures containg `[bass, drums, others, vocals]` source types,

This benchmark supports:
- Preparing the MUSDB dataset for **supervised training and evaluation**
- Running separation experiments using various discrete audio codecs and backbones (namely Conformer and CRDNN)
- Computing standard evaluation metrics (e.g., SDR, SIR, SAR) using the Fast-BSSEval library.

---

## Directory Structure

```
MUSDB
├── create_musdb.py # Generates chunked training data from raw MUSDB mixtures
├── README.md
├── separation
│   ├── musdb_prepare.py # Prepare MUSDB dataset for supervised SS training
│   ├── train.py # Unified training script for all MUSDB experiments
│   ├── utils.py # Audio I/O and utility functions
│   ├── hparams
│   │   ├── conformer
│   │   │   ├── train_dac.yaml # config recipe for Conformer
            ...
│   │   └── crdnn
│   │       ├── train_dac.yaml # config recipe for CRDNN
            ...
│   └── metrics
│       └── bsseval.py # BSSEval implementation (SDR, SIR, SAR)
└── experiments
```


---

## Setup

**Install dependencies:**

You may need additional packages for separation and evaluation:
```bash
pip install -r ../extra_requirements.txt
```

---

## Data Preparation

- Download the raw MUSDB dataset: Please follow the instructions from the [official project page](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) to download the data locally
- Unpack it into a directory `<musdb_dir>`
- To validate the data and create the MUSDB mixtures, you'll need to run `create_musdb.py`. That will create mixtures for all three sets, namely `['train']`. Note that this script only creates the `train` set and necessitate a `--num_chunks` argument which will define the number of random chunks to take per track. For `eval` and `validation`, please run `create_musdb_eval.py` instead.
- Lastly, to create the `.csv` manifests, run `separation/musdb_prepare.py`

## Running Separation Experiment

```python
python MUSDB/separation/train.py MUSBD/separation/hparams/conformer/train_encodec.yaml \
    --data_folder=<musdb_dir> \
    --output_folder=MUSDB/experiments
```