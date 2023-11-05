# DASB - Discrete Audio and Speech Benchmark

Work in progress
Please, install unstable-v0.6 version of SpeechBrain

This repository provides scripts for benchmarking discrete audio representations on various downstream tasks, including ASR, speaker verification, emotion recognition, and TTS.

Large Language Models are revolutionizing the field of artificial intelligence, advancing us towards artificial general intelligence. The dominant technology behind them is based on autoregressive transformer-based models that predict the next token (e.g. a character or a subword unit), based on the preceding ones. Language models use text data that are inherently discrete in nature.  Conversely, the domain of audio and speech processing predominantly relies on continuous representations, such as mel-spectrograms or popular self-supervised features like wav2vec2, wavLM, and Hubert. Interestingly, a notable trend towards discrete audio representations is gaining momentum, leading to the emergence of discrete audio tokens similar to those employed in language models. Leveraging audio tokens offers the potential to facilitate the creation of multimodal language models capable of processing not only text but also audio inputs.

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

To install and set up the benchmark, follow these steps:

1. Install SpeechBrain unstable-v0.6 branch :
   ```shell
   pip install git+https://github.com/speechbrain/speechbrain.git@unstable-v0.6
   ```

2. Clone the benchmark repository:
   ```shell
   git clone --branch DASB  https://github.com/poonehmousavi/SB_benchmarks.git
   ```

3. Navigate to `<path-to-repository>/benchmarks/DASB` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements.

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

### Running an experiment


---------------------------------------------------------------------------------------------------------
## 📈️ Results

---------------------------------------------------------------------------------------------------------
## 📧 Contact

[mousavi.pooneh@gmail.com](mailto:mousavi.pooneh@gmail.com)

---------------------------------------------------------------------------------------------------------
