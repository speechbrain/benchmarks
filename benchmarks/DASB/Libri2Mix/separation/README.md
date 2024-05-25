# Speech Separation with Discrete Audio Representations

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train speech separation systems based on discrete audio representations.

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

### LibriMix

Follow the instructions from the [official repository](https://github.com/JorisCos/LibriMix).

Expected folder structure: `LibriMix/Libri<N>Mix/wav16k/min/{train-100, train-360, dev, test}/{mix_both, mix_clean, noise, s1, ..., s<N>}`

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Open a terminal and run:

```bash
pip install -r extra-requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Open a terminal and run:

```bash
python train_<model>.py hparams/<path-to-config>.yaml --data_folder <path-to-data-folder>
```

### Examples

```bash
python train_encodec.py hparams/encodec.yaml --data_folder data/LibriMix
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
