# Speech Separation with Discrete Audio Representations

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train speech separation systems
based on discrete audio representations.

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

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
cd LibriMix/Transformer
python train_encodec.py hparams/train_encodec.yaml --data_folder data/LibriMix
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
