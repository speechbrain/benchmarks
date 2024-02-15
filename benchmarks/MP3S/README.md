# Multi-probe Speech Self Supervision Benchmark (MP3S)
This repository provides scripts for benchmarking self-supervised representations on various downstream tasks, including ASR, speaker verification, emotion recognition, and intent classification.

The key feature of this benchmark is that it allows users to choose their desired probing head for downstream training. This is why we called it the *Multi-probe Speech Self Supervision Benchmark (MP3S)*. It has been demonstrated that the performance of the model is greatly influenced by this selection. For more information about the motivation, tasks, and results, please refer to the following [paper](https://arxiv.org/abs/2306.00452):

```bibtex
@article{zaiem2023speech,
  title={Speech Self-Supervised Representation Benchmarking: Are We Doing it Right?},
  author={Zaiem, Salah and Kemiche, Youcef and Parcollet, Titouan and Essid, Slim and Ravanelli, Mirco},
  journal={In Proc. of Interspeech},
  year={2023}
}
```

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

2. Navigate to `<path-to-repository>/benchmarks/MP3S` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements (i.e., kenlm for the n-gram language model).


## ▶️ Quickstart

### Running a single task

To perform a downstream evaluation using a specific SSL model (e.g., `facebook/wav2vec2-base`) with one of the available probing heads (e.g., LSTM), run the following command:
   ```
   python LibriSpeech/LSTM/train.py LibriSpeech/LSTM/hparams/ssl.yaml --data_folder mypath/to/LibriSpeech
   ```

### Running a single task with your SSL model

If you have your own SSL model and want to benchmark it for a specific task, you need to follow these instructions:

1. Make sure your model is available on [HuggingFace](https://huggingface.co/) (our recipes will fetch it from the specified HF path).
2. Assume your model is located at `mygroup/mySSLModel` on [HuggingFace](https://huggingface.co/).
3. Run the following command:
   ```
   python LibriSpeech/LSTM/train.py LibriSpeech/LSTM/hparams/ssl.yaml --ssl_hub mygroup/mySSLModel --encoder_dim 768 --num_layers_ssl 13 --output_folder my-output-folder --data_folder mypath/to/LibriSpeech
   ```

### Running multiple tasks

To run all tasks, make the following changes:

1. Edit the `run_benchmark.sh` file and modify the SSL-related values. Specify the HF hub, the encoder dimension (size of each frame vector), and the number of layers.
2. Choose a set of tasks from the provided list and, for each task, select a downstream architecture from the available options (see list below).
3. Update the variables defined in `run_benchmark.sh` with two lists of equal size. In the `ConsideredTasks` list, specify the tasks you want to run (e.g., `'LibriSpeechASR' 'LibriSpeechASR' 'IEMOCAP'`). In the `Downstreams` list, specify the corresponding downstream architecture for each task (e.g., `'BiLSTM'`, `contextnet`, `'ecapa_tdnn'`).

   For example, if you set `ConsideredTasks=('LibriSpeechASR' 'LibriSpeechASR' 'IEMOCAP')` and `Downstreams=('BiLSTM', 'contextnet', 'ecapa_tdnn')`, the benchmark will be executed as follows:
   - LibriSpeechASR with BiLSTM as the probing head
   - LibriSpeechASR with contextnet as the probing head
   - IEMOCAP with ecapa_tdnn as the probing head.

## Tasks and downstream heads

| Dataset | Task                                     | First Probing Head    | Second Probing Head | Dataset Download                                         |
|----------|------------------------------------------|-----------------------|---------------------|----------------------------------------------------------|
| LibriSpeech |  Speech Recognition                         | BiLSTM                | Conformer           | https://www.openslr.org/12                               |
|  Buckeye | Speech Recognition                               | BiLSTM                | ContextNet          | https://buckeyecorpus.osu.edu/                           |
| CommonVoice 11.0 (Welsh (Cymraeg) and Basque (Euskera) |  Low-Resource  ASR             | BiLSTM                | Linear              | https://commonvoice.mozilla.org/en/datasets                       |
| VoxCeleb1 |  Automatic Speaker Verification | X-Vectors             | ECAPA-TDNN          | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html |
|  IEMOCAP | Emotion Recognition              | Time-Pooling + Linear | ECAPA-TDNN          | https://sail.usc.edu/iemocap/                            |
| SLURP | Intent Classification              | Time-Pooling + Linear | BiLSTM + Linear     | https://zenodo.org/record/4274930                        |

The n-gram-based language model used during decoding for the English ASR experiments can be downloaded here: https://www.openslr.org/resources/11/4-gram.arpa.gz.

After downloading, to use it, you need to specify its path in the `ngram_lm_path` variable and activate language modeling during decoding by passing `language_modelling` True.

## 📈️ Results

The following table shows the results obtained with 2 different SSL encoders, and with two different probing heads for every task. The full table can be found in the [paper](https://arxiv.org/abs/2306.00452) presenting the benchmark.

| Models/Tasks                    | SSL Params | LibriSpeech Train-clean-100 | Buckeye ASR |  Welsh | Basque  |  VoxCeleb1 |       IEMOCAP      | SLURP Scenario identification |
|---------------------------------|:----------:|:---------------------------:|:-----------:|:------:|:-------:|:----------:|:------------------:|:-----------------------------:|
| Evaluation Metrics              |            |             WER             |     WER     |   WER  |   WER   |     EER    |      Accuracy      |            Accuracy           |
| First Downstream Architectures  |            |            BiLSTM           |    BiLSTM   | BiLSTM |  BiLSTM |  X-Vectors | Time-pool + Linear |       Time-pool + Linear      |
| Data2vec Base                   |    93.8M   |             5.30            |    37.26    |  54.00 |  46.37  |    5.43    |        63.0        |              56.9             |
| Data2vec Large                  |   314.3M   |             3.10            |    22.63    |  44.32 |  38.23  |    4.89    |        64.1        |              69.8             |
| Second Downstream Architectures |            |          Conformer          |  ContextNet |   Lin  |   Lin   | ECAPA-TDNN |     ECAPA-TDNN     |        BiLSTM + Linear        |
| Data2vec Base                   |    93.8M   |             5.85            |    40.53    |  77.49 |  75.26  |    3.75    |         72         |              73.4             |
| Data2vec Large                  |   314.3M   |             3.43            |    25.26    |  69.09 |  63.31  |    2.67    |        71.3        |              79.9             |

All the experiments presented in this work, including those with large SSL encoders such as Data2Vec Large, should be able to run on a 32Gb V100 GPU with the batch sizes provided in the yaml files. These batch sizes may be increased for faster training with smaller "Base" SSL encoders.

The next table provides estimates of training time for every couple (task, probing head), using Data2vec Large as the SSL encoder. For more details, we provide also the [logs (and checkpoints!) of experiments](https://www.dropbox.com/sh/twwihgivj0e8tpd/AABukLIvt0rjy7ZOK5DZXjD0a?dl=0) done with Data2Vec Base and Large versions):

|                   Task                   | First Probing Head - Training Duration | Second Probing Head - Training Duration |                            [Data2Vec Base Logs](https://www.dropbox.com/sh/rknral0jub5vl4x/AABBN5mU7Xermrd_ICSXMS6La?dl=0)                            | [Data2Vec Large Logs](https://www.dropbox.com/sh/l4lh6ngv8rzcrnp/AADWLG97ifuCv65AnH-rg2_Qa?dl=0)                                                      |
|:----------------------------------------:|:--------------------------------------:|:---------------------------------------:|:------------------------------------------------------------------------:|--------------------------------------------------------------------------|
| LibriSpeech ASR                          | 12 hours                               | 23 hours                                | https://www.dropbox.com/sh/c494uhaxp8gpdi3/AABI2_-zenjObL8i0RmVAjzHa?dl=0 | https://www.dropbox.com/sh/rchaoc5kkkslp7a/AAA314zxBxBqN8vUSFLYF8zQa?dl=0 |
| Buckeye ASR                              | 6 hours                                | 7 hours                                 | https://www.dropbox.com/sh/aedpbio3wf0qlkb/AAD5fIaM-q6YWI31CMpNiGWNa?dl=0 | https://www.dropbox.com/sh/8p5s8ppjfr89g6f/AAAevLOybveHY5q8KPggxofCa?dl=0 |
| CommonVoice Low-Resource ASR             | 3h30                                   | 2 hours                                 | https://www.dropbox.com/sh/21j16d7pc8g19yv/AABufUR7W8qe2SUcPcs9CWIDa?dl=0 | https://www.dropbox.com/sh/7fga8zlr1ftpk4u/AACm6LEWm2uH-EiJfolcJ3iPa?dl=0 |
| VoxCeleb1 Automatic Speaker Verification | 23 hours                               | 25 hours                                | https://www.dropbox.com/sh/gqf4l4nivj84517/AACpe-dtnzYIkmrM0FGoe43Da?dl=0 | https://www.dropbox.com/sh/88itv8lfbqxjajd/AAAjbc-BSGVCO2bQOh9l8sToa?dl=0 |
| IEMOCAP Emotion Recognition              | 1h30                                   | 2 hours                                 | https://www.dropbox.com/sh/s75465qsszjlyfm/AADeLQEFGi6wrG7x6LX2uCJMa?dl=0 | https://www.dropbox.com/sh/zrmh4avv970ws28/AADJH1kGDBH6kqMsu970U0Jia?dl=0 |
| SLURP Intent Classification              | 8 hours                                | 9 hours                                 | https://www.dropbox.com/sh/g163esobi1bkjf1/AABqpUGX6k1Ydt5gZo6m7xKHa?dl=0 | https://www.dropbox.com/sh/itf1vxc5vxiw912/AADUwY2i8vlt5aBjBAKCfl1na?dl=0 |

## 📧 Contact

[zaiemsalah@gmail.com](mailto:zaiemsalah@gmail.com)

# **Citing**
If you use this MP3S benchmark, please cite :

```bibtex
@article{zaiem2023speech,
  title={Speech Self-Supervised Representation Benchmarking: Are We Doing it Right?},
  author={Zaiem, Salah and Kemiche, Youcef and Parcollet, Titouan and Essid, Slim and Ravanelli, Mirco},
  journal={arXiv preprint arXiv:2306.00452},
  year={2023}
}
```
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
