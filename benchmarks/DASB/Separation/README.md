# Source Separation with Discrete Speech Representations

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train source separation systems
based on discrete speech representations (for reference, see [TokenSplit](https://arxiv.org/abs/2308.10415)).

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

### WSJ0Mix

Download the WSJ0 dataset in `<path-to-data-folder>` from the [official website](https://catalog.ldc.upenn.edu/LDC93s6a).
The best way to create the `2mix` and `3mix` data is using the original MATLAB script. This script and the associated metadata can be
obtained [here](https://www.dropbox.com/s/gg524noqvfm1t7e/create_mixtures_wsj023mix.zip?dl=1).
The dataset creation script assumes the original WSJ0 files in `SPHERE` format are already converted to `WAV`.

### LibriMix

Follow the instructions from the [official repository](https://github.com/JorisCos/LibriMix).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Open a terminal and run:

```bash
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<dataset>/<probing-head>`, open a terminal and run:

```bash
python train_<variant>.py hparams/train_<variant>.yaml --data_folder <path-to-data-folder>
```

### Examples

```bash
cd WSJ0Mix/Transformer
python train_encodec.py hparams/train_encodec.yaml --data_folder data/wsj0-2mix-8k-min/wsj0-mix
```

```bash
cd LibriMix/RNN
python train_encodec.py hparams/train_encodec.yaml --data_folder data/LibriMix
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
