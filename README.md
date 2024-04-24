# SpeechBrain Benchmarks

<p align="center">
  <img src="https://raw.githubusercontent.com/speechbrain/speechbrain/develop/docs/images/speechbrain-logo.svg" alt="SpeechBrain Logo"/>
</p>

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/SpeechBrain1/)
[![Discord](https://dcbadge.vercel.app/api/server/3wYvAaz3Ck?style=flat)](https://discord.gg/3wYvAaz3Ck)

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star our project on github (see top-right corner) if you appreciate our contribution to the community!**

Welcome to the SpeechBrain Benchmarks repository! This repository is dedicated to housing a collection of benchmarks associated with the [SpeechBrain toolkit](https://speechbrain.github.io/).

*What are benchmarks?* Benchmarks are standardized sets of recipes that enable users to measure the performance of specific models or techniques within a standardized environment. By utilizing these benchmarks, you can evaluate and compare the effectiveness of different approaches.

The SpeechBrain Benchmarks currently include the following:

- [CL_MASR](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/CL_MASR) - A benchmark designed to assess continual learning techniques, specifically focusing on the continual learning of new languages for speech recognition.

- [MP3S](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MP3S) - A benchmark created to facilitate the fair assessment of self-supervised speech representations.

- [MOABB](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB) - A benchmark designed for evaluating neural models in well-known EEG tasks like motor imagery, P300, and SSVEP.


Our benchmarks align with speechbrain's commitment to transparent and replicable research by providing standardized tools and resources for studying specific topics.

To learn more about each available benchmark, please click on the corresponding link provided above. We encourage you to explore the benchmarks and utilize them for your research and experimentation.

If you have any questions or need assistance, please don't hesitate to reach out. Happy benchmarking!

## 🛠️ Installation

1. Clone the GitHub repository and install the requirements:

    ```bash
    git clone https://github.com/speechbrain/benchmarks.git
    cd benchmarks
    git submodule update --init --recursive
    cd speechbrain
    pip install -r requirements.txt
    pip install -e .
    ```

2. Access SpeechBrain in your Python code:

    ```python
    import speechbrain as sb
    ```

Any modifications made to the `speechbrain` package will be automatically reflected, thanks to the `--editable` flag.

## ✔️ Test Installation

Ensure your installation is correct by running the following commands:

```bash
pytest tests
pytest --doctest-modules speechbrain
```

## 🏃‍♂️ Running an Experiment

In SpeechBrain, you can train a model for any task using the following steps:

```python
cd recipes/<dataset>/<task>/
python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the YAML file.

## 📘 Learning SpeechBrain

- **Website:** Explore general information on the [official website](https://speechbrain.github.io).

- **Tutorials:** Start with [basic tutorials](https://speechbrain.github.io/tutorial_basics.html) covering fundamental functionalities. Find advanced tutorials and topics in the Tutorials menu on the [SpeechBrain website](https://speechbrain.github.io).

- **Documentation:** Detailed information on the SpeechBrain API, contribution guidelines, and code is available in the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html).


# Citing SpeechBrain
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


