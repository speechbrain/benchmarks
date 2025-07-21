# FUSS Separation Task

This folder defines the **FUSS source separation benchmark** within DASB (Discrete Audio Separation Benchmark). It enables evaluating discrete audio representations on **general-purpose source separation**, using the [FUSS dataset](https://www.tensorflow.org/datasets/catalog/fuss) (Free Universal Sound Separation).

## Overview

The goal of this task is to perform **source separation** from complex acoustic mixtures of general sounds and go beyond speech and music,

This benchmark supports:
- Preparing the FUSS dataset for **supervised training and evaluation**
- Running separation experiments using various discrete audio codecs and backbones (namely Conformer and CRDNN)
- Computing standard evaluation metrics (e.g., SDR) using the Fast-BSSEval library.

---

## Directory Structure

```
FUSS
├── create_fuss.py # Generates chunked training data from raw FUSS mixtures
├── README.md
├── separation
│   ├── fuss_prepare.py # Prepare FUSS dataset for supervised SS training
│   ├── train.py # Unified training script for all FUSS experiments
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

- Download the raw FUSS dataset: Please follow the instructions from the [official repo](https://github.com/google-research/sound-separation/tree/master/datasets/fuss) to download the data locally
- Unpack it into a directory `<fuss_dir>`
- To validate the data and create the FUSS mixtures, you'll need to run `create_fuss.py`. That will create mixtures for all three sets, namely `['eval', 'train', 'validation']`
- Lastly, to create the `.csv` manifests, run `separation/fuss_prepare.py`

## Running Separation Experiment

```python
python FUSS/separation/train.py FUSS/separation/hparams/conformer/train_encodec.yaml \
    --data_folder=<fuss_dir> \
    --output_folder=FUSS/experiments
```