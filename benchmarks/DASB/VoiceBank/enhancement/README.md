# Speech Enhancement with Discrete Audio Representations

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train speech enhancement systems based on discrete audio representations.

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

### VoiceBank

Download the following files from the [official website](https://datashare.ed.ac.uk/handle/10283/2791):

- `clean_testset_wav.zip`
- `clean_trainset_28spk_wav.zip`
- `noisy_testset_wav.zip`
- `noisy_trainset_28spk_wav.zip`

Extract them to a folder named `VoiceBank`.

Expected folder structure: `VoiceBank/{clean_testset_wav, clean_trainset_28spk_wav, noisy_testset_wav, noisy_trainset_28spk_wav}`

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
python train_encodec.py hparams/conformer/encodec.yaml --data_folder data/VoiceBank
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
