
```diff
- ***********************************  Work in progress ***********************************
```
@@ text in purple (and bold)@@
# Discrete Audio and Speech Benchmark (MDASB)
This repository provides scripts for benchmarking discrete representations on various downstream tasks, including ASR, speaker verification, emotion recognition, intent classification, keyword spotting, enhancement, separation and TTS .

The key feature of this benchmark .....



## 🛠️️ Installation

To set up the benchmark, follow these steps:


1. Clone the benchmark repository and install SpeechBrain:
   ```shell
   git clone https://github.com/speechbrain/benchmarks.git
   cd benchmarks
   git submodule update --init --recursive
   cd speechbrain
   pip install -r requirements.txt
   pip install -e .
   ```

2. Navigate to `<path-to-repository>/benchmarks/DASB` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements (i.e., kenlm for the n-gram language model).


## ▶️ Quickstart

### Running a single task

To perform a downstream evaluation using a specific discrete  model (e.g., encodec), run the following command:
   ```
   TODO
   ```

### Running a single task with your SSL model

If you have your own discrete model and want to benchmark it for a specific task, you need to follow these instructions:

1. Make sure your model is available on [HuggingFace](https://huggingface.co/) (our recipes will fetch it from the specified HF path).
2. Assume your model is located at `mygroup/mySSLModel` on [HuggingFace](https://huggingface.co/).
3. Run the following command:
   ```
   TODO
   ```

### Running multiple tasks



## Tasks and downstream heads

| Dataset | Task                                      | Model | Dataset Download                                         |
|----------|------------------------------------------|-----------------------|----------------------------------------------------------|
| LibriSpeech |  Speech Recognition                         | Conformer           | https://www.openslr.org/12                               |
| CommonVoice 14.0 (multilingual) |  Low-Resource  ASR             | Conformer              | https://commonvoice.mozilla.org/en/datasets                       |
| VoxCeleb1 |  Automatic Speaker Verification | ECAPA-TDNN          | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html |
|  IEMOCAP | Emotion Recognition             | ECAPA-TDNN          | https://sail.usc.edu/iemocap/                            |
| SLURP | Intent Classification              | BiLSTM + Linear     | https://zenodo.org/record/4274930                        |
|Google-speech-commands | Keyword Spotting |      ECAPA-TDNN        |https://research.google/blog/launching-the-speech-commands-dataset/ |
|VoiceBank | Enhancement |      ...       |..... |
|.... | Separation |      ...       |..... |
|.... | TTS-single speaker |      ...       |..... |
|.... | TTS-multi spekaer speaker |      ...       |..... |
|.... | Vocoder-single spekaer speaker |      ...       |..... |
|.... | Vocoder-multi spekaer speaker |      ...       |..... |





The n-gram-based language model used during decoding for the English ASR experiments can be downloaded here: https://www.openslr.org/resources/11/4-gram.arpa.gz.

After downloading, to use it, you need to specify its path in the `ngram_lm_path` variable and activate language modeling during decoding by passing `language_modelling` True.

## 📈️ Results

The following table shows the results obtained .....


## 📧 Contact

[mousavi.pooneh@gmail.com](mailto:mousavi.pooneh@gmail.com)

# **Citing**
If you use this DASB benchmark, please cite :


Please, cite SpeechBrain if you use it for your research or business.

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
