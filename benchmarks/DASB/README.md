<p align="center">
  <img src="DASB_logo.png" alt="SpeechBrain-DASB logo" width=200/>
</p>

# DASB - Discrete Audio and Speech Benchmark

This repository provides a benchmark for evaluating discrete audio representations using popular audio tokenizers like **EnCodec**, **DAC**, and many more, integrated with SpeechBrain.

The package helps integrate and evaluate new audio tokenizers in speech tasks of great interest such as *speech recognition*, *speaker identification*, *emotion recognition*, *keyword spotting*, *intent classification*, *speech enhancement*, *separation*, and *text-to-speech*. It offers an interface for easy model integration and testing and a protocol for comparing different audio tokenizers.

This repository can be used to benchmark new audio tokenizers reliably. It includes a benchmark on 9 audio and speech datasets using 6 popular discrete audio encoders: **semantic** (*Discrete HuBERT*, *Discrete WavLM*, *Discrete Wav2Vec2*), **compression** (*EnCodec*, *DAC*), and **hybrid** (*SpeechTokenizer*). We consider different downstream architectures for each task and report the best-performing architecture.

For detailed information, refer to [paper](https://arxiv.org/pdf/2406.14294):

```bibtex
@misc{mousavi2024dasb,
  title={DASB - Discrete Audio and Speech Benchmark},
  author={Pooneh Mousavi and Luca Della Libera and Jarod Duret and Artem Ploujnikov and Cem Subakan and Mirco Ravanelli},
  year={2024},
  eprint={2406.14294},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#-installation)
- [Discrete Audio Encoder](#-Discrete-Audio-Encoder)
- [Datasets and Recipes](#-Datasets-and-Recipes)
- [Quickstart](#-quickstart)
  - [Running a single task](#Running-a-single-task)
  - [Running multiple tasks](#Runnin-multiple-tasks)
- [‍Incorporating Your Audio Tokenizer](#-Incorporating-Your-Audio-Tokenizer)
- [Results](#-results)
- [Contact](#-contact)
- [Citing](#-citing)

# 🛠️ Installation

To set up SpeechBrain-DASB, follow these steps:


1. Clone the benchmark repository and install SpeechBrain:
   ```shell
   git clone --branch DASB https://github.com/speechbrain/benchmarks.git
   cd benchmarks
   git clone https://github.com/speechbrain/speechbrain/
   cd speechbrain
   pip install -r requirements.txt
   pip install -e .
   ```

2. Navigate to `<path-to-repository>/benchmarks/DASB` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements.

3. Install the pretrained UTMOS model. This step is only required if you want to run TTS experiments.

    Ensure that you have the git-lfs extension installed: https://git-lfs.com. It is needed to automatically
      download pretrained checkpoint files.

      ```shell
      cd path/to/utmos
      git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo utmos
      cd utmos
      pip install -r requirements.txt
      git lfs fetch --all
      ```

# 🎌 Discrete Audio Encoder
| **Model**                                                                 | **Dataset**                                                                                       | **Repo**                                                         |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [Discrete Hubert](https://arxiv.org/abs/2406.10735)                      | LibriSpeech960                                                                                     | [huggingface.co/speechbrain/SSL_Quantization](https://huggingface.co/poonehmousavi/SSL_Quantization)  |
| [Discrete WavLM](https://arxiv.org/abs/2406.10735)                       | LibriSpeech960                                                                                     | [huggingface.co/speechbrain/SSL_Quantization](https://huggingface.co/poonehmousavi/SSL_Quantization)  |
| [Discrete Wav2Vec2](https://arxiv.org/abs/2406.10735)                    | LibriSpeech960                                                                                     | [huggingface.co/speechbrain/SSL_Quantization](https://huggingface.co/poonehmousavi/SSL_Quantization)  |
| [EnCodec](https://arxiv.org/abs/2210.13438)                              | DNS, CommonVoice, AudioSet, FSD50K, and Jamendo                                                    | [github.com/facebookresearch/encodec](https://github.com/facebookresearch/encodec)                    |
| [DAC](https://arxiv.org/abs/2306.06546)                                  | DAPS, DNS, CommonVoice, VCTK, MUSDB, and Jamendo                                                   | [github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)     |
| [SpeechTokenizer](https://arxiv.org/abs/2308.16692)                      | LibriSpeech960                                                                                     | [github.com/ZhangXInFD/SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)                 |

# ⚡ Datasets and Recipes

| **Dataset**                              | **Task**                             | **1st Architecture**        | **2nd Architecture**          | **Dataset Link**                                                                       |
|------------------------------------------|--------------------------------------|-----------------------------|-------------------------------|----------------------------------------------------------------------------------------|
| LibriSpeech                              | Speech Recognition                   | BiLSTM                      | ContextNet                    | [openslr.org/12](https://openslr.org/12)                                               |
| CommonVoice 17.0                         | Speech Recognition                   | BiLSTM                      | Linear                        | [commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)     |
| VoxCeleb1                                | Speaker Verification/Identification  | ECAPA-TDNN                  | X-Vectors                     | [robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) |
| IEMOCAP                                  | Emotion Recognition                  | ECAPA-TDNN                  | Time-Pooling + Linear         | [sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)                                |
| Speech Commands                          | Keyword Spotting                     | X-Vectors                   | ECAPA-TDNN                    | [tensorflow.org/datasets/catalog/speech_commands](https://www.tensorflow.org/datasets/catalog/speech_commands) |
| SLURP                                    | Intent Classification                | BiLSTM + Linear             | Time-Pooling + Linear         | [zenodo.org/record/4274930](https://zenodo.org/record/4274930)                        |
| VoiceBank                                | Speech Enhancement                   | Conformer                   | CRDNN                         | [datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)  |
| Libri2Mix                                | Speech Separation                    | Conformer                   | CRDNN                         | [github.com/JorisCos/LibriMix](https://github.com/JorisCos/LibriMix)                  |
| LJSpeech                                 | Text-to-Speech                       | Shallow Transformer         | Deep Transformer              | [keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)            |

# ▶️ Quickstart

## Running a single task

If you have specific discrete model and want to benchmark it for a specific task, you need to run the following command:
   ```
   python LibriSpeech/ASR/LSTM/train_[tokenzier_name].py LibriSpeech/ASR/LSTM/hparams/train_[tokenzier_name].yaml --output_folder my-output-folder --data_folder mypath/to/LibriSpeech
   ```

## Running multiple tasks

To run all tasks, make the following changes:

1. Edit the `run_discriminative_benchmark.sh` and `run_genarative_benchmark.sh` files and modify tokenizer related values for example the bitrate , number of codebooks, and etc.
2. Choose a set of tasks from the provided list and, for each task, select a downstream architecture from the available options (see list below).
3. Update the variables defined in `run_benchmark.sh` with two lists of equal size. In the `ConsideredTasks` list, specify the tasks you want to run (e.g., `'LibriSpeechASR' 'LibriSpeechASR' 'IEMOCAP'`). In the `Downstreams` list, specify the corresponding downstream architecture for each task (e.g., `'BiLSTM'`, `contextnet`, `'ecapa_tdnn'`).

   For example, if you set `ConsideredTasks=('LibriSpeechASR' 'LibriSpeechASR' 'IEMOCAP')` and `Downstreams=('BiLSTM', 'contextnet', 'ecapa_tdnn')`, the benchmark will be executed as follows:
   - LibriSpeechASR with BiLSTM as the probing head
   - LibriSpeechASR with contextnet as the probing head
   - IEMOCAP with ecapa_tdnn as the probing head.

3. Run the following command:
   ```
   bash run_discriminative_benchmark.sh [tokenzier_name]
   bash run_genarative_benchmark.sh [tokenzier_name]
   ```
   You could also pass extra arguments as far as they are consistent  across all tasks.

   For generative task, make sure to set the `utmos_path` required for TTS evaluation.
# 📝 ‍Incorporating Your Audio Tokenizer

Let's now assume you've designed an audio and speech tokenizer in PyTorch and wish to integrate it into our benchmark.
You're in luck because we've made this step as simple as possible for you!
Here are the steps you should follow:

1. Write your model's code in a Python library saved in `benchmarks/DASB/model` (e.g., `benchmarks/MOABB/models/my_model.py`).

2. Create a YAML and py file for each task you want to experiment with. Thankfully, you don't have to start from scratch. For example, if you're working with LibriSpeech/ASR/LSTM, copy `benchmarks/DASB/LibriSpeech/ASR/contextnet/hparams/train_encodec.yaml` and save it in the same folder with a different name (e.g., `train_my_model.yaml` and `train_my_model.py`).

3. Edit the relevant section of your `train_my_model.yaml` and `train_my_model.py`. Redefine the `codec:` to reference your custom model (e.g., `codec: !new:models.my_model.my_model`).

4. Ensure you include the hyperparameters specific to your model.

5. Now, follow the instructions above to run an experiments across tasks.
**Note**: If you're not familiar with YAML, you can refer to our [HyperPyYAML tutorial](https://speechbrain.github.io/tutorial_basics.html) on the SpeechBrain website for guidance.

# 📈 Results
We present the performance results for discriminative and generative tasks using two different downstream architectures. Each value in the table reflects the best performance achieved with either architecture. We also categorize the results into three distinct bitrate ranges: low (0-1.5 kbps), medium (2.9-6 kbps), and high (24 kbps).
## Ranking
We present the ranking aggregation for the considered tokenizers (medium bitrate) sorted by "Combination" column. Each model is individually ranked for every task, and we compute the average position across all ranks.
| Model             | Discriminative | Generative | Combination |
|-------------------|----------------|------------|-------------|
| Discrete WavLM    | **2.00**       | 2.75       | **1.94**    |
| Discrete HuBERT   | 2.66           | 3.62       | 3.11        |
| SpeechTokenizer   | 3.44           | 3.81       | 3.64        |
| Discrete Wav2Vec2 | 3.33           | **2.68**   | 3.41        |
| EnCodec           | 4.11           | 3.93       | 4.23        |
| DAC               | 5.55           | 4.06       | 4.64        |

## Benchmarking Results for Discriminative Tasks

| **Models/Tasks**        | **ASR-En (WER ↓)** |               | **ASR-multiling (WER ↓)** |                     | **ER (ACC ↑)** | **IC (ACC ↑)** | **KS (ACC ↑)** | **SI (ACC ↑)** | **SV (EER ↓)** |
|-------------------------|--------------------|---------------|---------------------|---------------------------|----------------|----------------|----------------|----------------|----------------|
|                         | **Clean**          | **Other**     | **Welsh**                 | **Basque**                |                |                |                |                |                |
|          |                    |               |                           |      <span style="background-color: #FFD700; color: black;">**Low Bitrate**</span>             |                |                |                |                |                |
| Discrete Hubert         | **8.99**           | **21.14**     | **58.50**                 | **26.83**     | 57.20          | 68.70          | 90.54          | 0.90           | 24.99          |
| Discrete WavLM          | 11.72              | 27.56         | 60.37                     | 28.63         | **59.80**      | 73.40          | **97.94**      | 0.70           | 26.02          |
| Discrete Wav2Vec2       | 12.14              | 28.65         | 66.30                     | 32.25         | 57.80          | **74.10**      | 96.16          | 0.40           | 33.53          |
| EnCodec                 | 52.37              | 77.04         | 92.01                     | 58.20         | 44.70          | 31.50          | 86.00          | **58.30**      | **17.40**      |
| DAC                     | 63.96              | 83.61         | 94.86                     | 66.29         | 49.20          | 22.10          | 81.00          | 45.10          | 20.62          |
| SpeechTokenizer         | 19.77              | 43.12         | 76.67                     | 47.92         | 49.10          | 57.90          | 95.09          | 47.40          | 20.41          |
|          |                    |               |                           |      <span style="background-color: #00BFFF; color: black;">**Medium Bitrate**</span>             |                |                |                |                |
| Discrete Hubert         | **7.91**           | **18.95**     | 54.77                     | 23.63         | **62.10**      | 70.50          | 94.69          | 67.40          | 15.71          |
| Discrete WavLM          | 8.52               | 20.35         | **54.22**                 | **22.06**     | 57.60          | **78.00**      | **98.09**      | 80.80          | **8.00**       |
| Discrete Wav2Vec2       | 8.76               | 21.32         | 60.39                     | 26.64         | 59.10          | 75.10          | 96.64          | 65.47          | 17.64          |
| EnCodec                 | 46.80              | 74.24         | 91.23                     | 47.95         | 51.30          | 31.40          | 88.70          | **91.90**      | **7.81**       |
| DAC                     | 59.54              | 81.48         | 97.43                     | 56.16         | 45.80          | 18.90          | 76.60          | 83.80          | 11.78          |
| SpeechTokenizer         | 18.32              | 41.21         | 75.17                     | 38.94         | 52.10          | 57.80          | 94.86          | 91.40          | 7.88           |
|          |                    |               |                           |      <span style="background-color: #32CD32; color: black;">**High Bitrate**</span>               |                |                |                |                |                |
| EnCodec                 | **45.18**          | **72.56**     | **93.40**                 | **87.65**     | 46.40          | **19.60**      | **83.60**      | **92.81**      | **7.18**       |
| DAC                     | 99.53              | 99.38         | 99.40                     | 99.68         | **46.00**      | 15.70          | 75.20          | 85.61          | 10.89          |
|          |                    |               |                           |      <span style="background-color: #FF4500; color: black;">**Continuous Baseline**</span>               |                |                |                |                |                |
| SSL                     | 3.370              | 7.04          | 41.77                     | 14.32         | 63.10          | 86.10          | 99.00          | 99.70          | 2.10           |

## Benchmarking Results for Generative Tasks


| **Models/Tasks**        | **SE**            |              |              | **SS**            |              |              | **TTS**           |              |
|-------------------------|-------------------|--------------|--------------|------------------|--------------|--------------|-------------------|--------------|
|                         | **DNSMOS ↑**      | **dWER ↓**   | **SpkSim ↑** | **DNSMOS ↑**     | **dWER ↓**   | **SpkSim ↑** | **UTMOS ↑**       | **dWER ↓**   |
|                     |             |           |             |             |           |             |             |            |
|         |                   |              |              |      <span style="background-color: #FFD700; color: black;">**Low Bitrate**</span>             |              |              |                   |              |
| Discrete HuBERT         | 3.33              | **15.47**    | 0.824        | 3.52             | 80.86        | 0.840        | 3.24              | **2.55**     |
| Discrete WavLM          | 3.26              | 16.52        | 0.830        | 3.43             | **62.34**    | 0.847        | **3.84**          | 3.01         |
| Discrete Wav2Vec2       | **3.55**          | 18.86        | 0.779        | **3.75**         | 96.70        | 0.787        | 3.32              | 3.45         |
| EnCodec                 | 3.15              | 34.35        | 0.852        | 3.11             | 83.55        | **0.877**    | 1.46              | 8.85         |
| DAC                     | 3.30              | 57.41        | 0.853        | 3.01             | 102.00       | 0.854        | 1.97              | 10.68        |
| SpeechTokenizer         | 3.18              | 30.13        | **0.858**    | 3.13             | 85.25        | 0.874        | 2.51              | 3.69         |
|         |                   |              |              |      <span style="background-color: #00BFFF; color: black;">**Medium Bitrate**</span>              |              |              |                   |              |
| Discrete HuBERT         | 3.48              | 12.62        | 0.875        | 3.70             | 66.29        | 0.891        | 3.80              | 3.40         |
| Discrete WavLM          | 3.48              | **10.18**    | 0.889        | 3.68             | **34.03**    | 0.912        | **3.82**          | **2.45**     |
| Discrete Wav2Vec2       | **3.54**          | 17.60        | 0.858        | **3.75**         | 78.42        | 0.866        | 3.68              | 2.89         |
| EnCodec                 | 3.10              | 19.07        | 0.885        | 3.09             | 48.57        | 0.906        | 1.50              | 9.46         |
| DAC                     | 3.49              | 31.14        | **0.906**    | 3.26             | 55.43        | **0.924**    | 1.71              | 71.26        |
| SpeechTokenizer         | 3.49              | 23.44        | 0.876        | 3.42             | 60.75        | 0.906        | 1.96              | 53.26        |
|         |                   |              |              |      <span style="background-color: #32CD32; color: black;">**High Bitrate**</span>               |              |              |
| EnCodec                 | 2.87              | 68.22        | 0.814        | **2.95**         | **97.73**    | **0.839**    | N.C               | N.C          |
| DAC                     | **2.95**          | **46.07**    | **0.860**    | 2.53             | 208          | 0.784        | N.C               | N.C          |
|         |                   |              |              |      <span style="background-color: #FF4500; color: black;">**Continuous Baseline**</span>               |              |              |
| SSL                     | 3.49              | 4.92         | 0.928        | 3.68             | 9.97         | 0.939        | 3.71              | 2.94         |


You can find the full experiment folder (i.e., checkpoints, logs, etc) [here](https://www.dropbox.com/scl/fo/qoo5ofhqc2x2jdkr1veoh/ABm2CXF6Q7_0pL7fEXMtTqQ?rlkey=78a1mmxnvz7ta8r6i43qpr55c&st=pt6belnn&dl=0).

# 📧 Contact

For any questions or inquiries, feel free to reach Pooneh Mousavi, Ph.D. Student, Concordia/Mila University([mousavi.pooneh@gmail.com](mailto:mousavi.pooneh@gmail.com)).
<!-- ############################################################################################################### -->
# 📖 Citing

If you use SpeechBrain-DASB for your research or business, please cite:

```bibtex
@misc{mousavi2024dasb,
  title={DASB - Discrete Audio and Speech Benchmark},
  author={Pooneh Mousavi and Luca Della Libera and Jarod Duret and Artem Ploujnikov and Cem Subakan and Mirco Ravanelli},
  year={2024},
  eprint={2406.14294},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```

```bibtex
@inproceedings{mousavi2024,
  title={How Should We Extract Discrete Audio Tokens from Self-Supervised Models?},
  author={Mousavi, Pooneh and Duret, Jarod and Zaiem, Salah and Della Libera, Luca and Ploujnikov, Artem and Subakan, Cem and Ravanelli, Mirco},
  booktitle={Proceedings of Interspeech},
  year={2024},
}
```

Please also cite SpeechBrain:

```bibtex

@article{ravanelli2024open,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Ravanelli, Mirco and Parcollet, Titouan and Moumen, Adel and de Langen, Sylvain and Subakan, Cem and Plantinga, Peter and Wang, Yingzhi and Mousavi, Pooneh and Della Libera, Luca and Ploujnikov, Artem and others},
  journal={arXiv preprint arXiv:2407.00463},
  year={2024}
}

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

