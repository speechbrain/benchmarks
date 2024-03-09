# Semantic Token Tuning



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

2. Navigate to `<path-to-repository>/benchmarks/STT` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements (i.e., kenlm for the n-gram language model).


## ▶️ Quickstart

### Running a single task



### Running a single task with your SSL model




## Tasks and downstream heads



## 📈️ Results



## 📧 Contact

[mousavi.pooneh@gmail.com](mailto:mousavi.pooneh@gmail.com)

# **Citing**
If you use this MP3S benchmark, please cite :

```bibtex

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
