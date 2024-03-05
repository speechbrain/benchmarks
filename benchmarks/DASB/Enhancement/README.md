# Speech Enhancement with Semantic Tokens

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train speech enhancement systems based on semantic tokens.

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

### LJSpeechWHAM

Download `LJSpeechWHAM.tar.gz` from OVHcloud and extract it to a folder of your choice (e.g. `data`).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Open a terminal and run:

```bash
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `LJSpeechWHAM`, open a terminal and run:

```bash
python train_discrete.py hparams/<config>.yaml --data_folder <path-to-data-folder>
```

### Examples

```bash
cd LJSpeechWHAM
python train_discrete.py hparams/transformer_wavlm_ljspeech_ljspeech.yaml --data_folder data/LJSpeechWHAM
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
