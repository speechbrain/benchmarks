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

Here’s the updated **Table of Contents** for your GitHub README with corrections and better alignment:

---

# 📑 Table of Contents

- [DASB - Discrete Audio and Speech Benchmark](#dasb---discrete-audio-and-speech-benchmark)
- [🛠️ Installation](#-installation)
- [🎌 Discrete Audio Encoder](#-discrete-audio-encoder)
- [⚡ Datasets and Recipes](#-datasets-and-recipes)
- [📖 Training Scenarios](#-training-scenarios)
  - [On-the-Fly Token Extraction](#on-the-fly-token-extraction)
  - [Offline Token Extraction](#offline-token-extraction)
- [🎛️ Hyperparameter Tuning](#%EF%B8%8F-hyperparameter-tuning)
- [📝 Incorporating Your Audio Tokenizer](#-incorporating-your-audio-tokenizer)
- [📈 Results](#-results)
  - [Ranking](#ranking)
  - [Benchmarking Results for Discriminative Tasks](#benchmarking-results-for-discriminative-tasks)
  - [Benchmarking Results for Generative Tasks](#benchmarking-results-for-generative-tasks)
- [📧 Contact](#-contact)
- [📖 Citing](#-citing)

---

This structure provides a clear and logical flow, ensuring users can easily navigate the document. Each major section is linked appropriately, with sub-sections for detailed content. Let me know if additional adjustments are required!

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

# 📖 Training Scenarios

We offer two different training scenarios: **on-the-fly token extraction** and **offline token extraction**.

## On-the-Fly Token Extraction
In this scenario, audio tokens are extracted dynamically during training. To enhance efficiency, we use a caching mechanism where tokens are saved in memory during the first epoch and retrieved for subsequent epochs. However, this approach has some limitations:
- It works best when the dataset is small, the bitrate is low, and batching is sorted (not random).
- It is unsuitable when data augmentation is required.

You can also disable the caching mechanism if needed.

Currently, the on-the-fly token extraction is applied only in the recipe located at:
`LibriSpeech/ASR-on-the-fly`

If you wish to adapt this strategy for your own recipe, you can copy and modify the existing recipe as needed. Here's how to run the on-the-fly recipe:

```bash
python LibriSpeech/ASR-on-the-fly/train.py LibriSpeech/ASR-on-the-fly/hparams/LSTM/{TOKENIZER}.yaml --data_folder=path/LibriSpeech --output_folder=path/results/LibriSpeech/ASR/{TOKENIZER}/LSTM
```

> **Note:** On-the-fly extraction can be time-consuming, which is why we also provide an alternative approach: **offline token extraction**.


## Offline Token Extraction
In this scenario, all tokens are pre-extracted in a separate recipe. We recommend using the highest number of codebooks available for token extraction and then choosing the desired settings during training.

### Token Extraction Command
To extract tokens, use the following command:

```bash
python LibriSpeech/extraction/extract.py benchmarks/DASB/LibriSpeech/extraction/hparams/{tokenizer}.yaml --data_folder=path/LibriSpeech --num_codebooks=32
```

If you wish to initialize your embedding layer with the tokenizer's embeddings while training your downstream model, set the flag `save_embedding` to `True`. For discrete SSL tokenizers, you can specify a list of layers for `--num_codebooks` instead of a single number (e.g., `--num_codebooks=[3,7,12]`).

### Training with Pre-Extracted Tokens
Once tokens are extracted and saved, you can train a downstream model using the following command:

```bash
bash run_experiments.sh --hparams benchmarks/DASB/LibriSpeech/ASR/hparams/LSTM/train.yaml --data_folder LibriSpeech --cached_data_folder cache/ --output_folder results/LibriSpeech/ASR/encodec/LSTM --task ASR --dataset LibriSpeech --seed 1986 --nruns 2 --eval_metric WER --tokens_folder LibriSpeech/extraction-emb/speech_tokenizer/save/librispeech/
```

---

This workflow ensures flexibility, efficiency, and reproducibility for both training scenarios. Adapt the recipes as needed for your specific requirements!

Here's a polished and formatted version for clarity, suitable for a README or documentation:


# 🎛️ Hyperparameter Tuning

Efficient hyperparameter tuning is critical when introducing novel models or experimenting with diverse datasets. Our benchmark establishes a standardized protocol for hyperparameter tuning, leveraging [Orion](https://orion.readthedocs.io/en/stable/) to ensure fair and consistent model comparisons.

---

## **Overview**

Hyperparameter tuning is managed using the `./run_hparam_optimization.sh` script. This script coordinates multiple hyperparameter trials via `run_experiments.sh`.



## **Incorporating Orion Flags in Hparam Files**

To enable tuning, Orion flags should be directly embedded in the YAML hparam file using comments. For example, to optimize the learning rate (`lr`) parameter within a defined range, include the following line in the YAML file:

```yaml
lr_model: 0.0001 # @orion_step1: --lr_model~"loguniform(0.00001,0.5)"
```



## **Workflow of the Script**

The script operates as follows:

1. **Scans** the YAML hparam file for Orion flags.
2. **Executes** hyperparameter tuning using the `orion-hunt` command.
3. **Saves** the best hyperparameters for reference via `torch-info`.
4. **Iterates** until encountering flags such as `@orion_step<stepid>` in the YAML file.



## **Running Hyperparameter Optimization**

You can perform hyperparameter optimization using a command like this:

```bash
bash run_hparam_optimization.sh \
  --exp_name 'ASR-encodec-LSTM_hopt' \
  --hparams LibriSpeech/ASR/hparams/LSTM/train.yaml \
  --data_folder path/LibriSpeech \
  --cached_data_folder path/cache/ \
  --output_folder results/LibriSpeech/ASR/encodec/LSTM \
  --task ASR \
  --dataset LibriSpeech \
  --seed 1986 \
  --nruns 1 \
  --nruns_eval 5 \
  --eval_metric WER \
  --exp_max_trials 50 \
  --tokens_folder results/LibriSpeech/extraction-emb/encodec/save/librispeech/ \
  --run_name encodec
```

For more details on the arguments and customization options, refer to `./run_hparam_optimization.sh`.


### **Notes**

1. **Execution Time**:
   - Hyperparameter tuning may take several hours or even days, depending on the model complexity and dataset.

2. **GPU vs. CPU**:
   - By default, models are trained on GPU. To train on CPU instead, include the `--device cpu` flag.

3. **Monitoring Progress**:
   - Use the following command to monitor optimization status:
     ```bash
     orion status --all
     ```
   - Ensure that Orion-specific environment variables are set in your bash environment. For example:
     ```bash
     export ORION_DB_ADDRESS=results/LibriSpeech/ASR/encodec/LSTM/hopt/ASR-encodec-LSTM_hopt.pkl
     export ORION_DB_TYPE=pickleddb
     ```
     Adjust `ORION_DB_ADDRESS` according to your experiment.

4. **Resuming Optimization**:
   - You can interrupt the script at any point. It will resume from the last completed trial.

5. **Repetition of Optimization**:
   - For multiple repetitions of the same hyperparameter optimization, modify the `--exp_name` parameter.

6. **System Requirements**:
   - The script is designed for Linux-based systems. A bash script is provided instead of Python due to its ability to manage diverse training loops across various subjects and sessions.

---

This protocol ensures fair model comparison across diverse tasks and datasets. All reported results are derived using this standardized hyperparameter tuning methodology, enabling consistent assessments across models.

# 📝 ‍Incorporating Your Audio Tokenizer

Let's now assume you've designed an audio and speech tokenizer in PyTorch and wish to integrate it into our benchmark.
You're in luck because we've made this step as simple as possible for you!
Here are the steps you should follow:


1. Write your model's code in a Python library saved in `benchmarks/DASB/model` (e.g., `benchmarks/DASB/models/my_model.py`).

2. Add the tokenizer to `utils/tokenizer_interface.py` and ensure the `encode` and `decode` functions are consistent in functionality and output shape with the other tokenizers.

3. Create a YAML and Python file for each task you want to experiment with. Thankfully, you don't have to start from scratch. For example, you can copy `LibriSpeech/extraction/hparams/encodec.yaml`, adapt it based on your needs, and save it in the same folder with a different name (e.g., `LibriSpeech/extraction/hparams/{YOUR_TOKENIZER_NAME}.yaml`).

4. Edit the relevant sections of your `{YOUR_TOKENIZER_NAME}.yaml`. Redefine the `tokenizer:` field to reference your custom model (e.g., `tokenizer: !new:tokenizer_interface.your_tokenizer`).

5. Ensure you include the hyperparameters specific to your model.

6. Now, follow the instructions provided earlier to run experiments across tasks.
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

