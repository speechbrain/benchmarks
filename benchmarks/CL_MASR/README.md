# CL-MASR: A Continual Learning Benchmark for Multilingual ASR

This is the official benchmark platform accompanying the paper [CL-MASR: A Continual Learning Benchmark for Multilingual ASR](https://arxiv.org/abs/2310.16931).

It includes scripts to train [Whisper](https://cdn.openai.com/papers/whisper.pdf) and [WavLM](https://arxiv.org/abs/2110.13900)-based ASR systems
on a subset of 20 languages selected from [Common Voice 13](https://commonvoice.mozilla.org/en/datasets) in a continual learning fashion using a
handful of methods including rehearsal-based, architecture-based, and regularization-based approaches.

The goal is to continually learn new languages while limiting forgetting the previously learned ones.
An ideal method should achieve both positive forward transfer (i.e. improve performance on new tasks leveraging
shared knowledge from previous tasks) and positive backward transfer (i.e. improve performance on previous tasks
leveraging shared knowledge from new tasks).

The following algorithms have been implemented so far:

- **Rehearsal-based**
  - [Experience Replay (ER)](https://arxiv.org/abs/1811.11682)
  - [Averaged Gradient Episodic Memory (A-GEM)](https://arxiv.org/abs/1812.00420)
  - [Dark Experience Replay (DER)](https://arxiv.org/abs/2004.07211) (task-incremental variant)

- **Architecture-based**
  - [Progressive Neural Networks (PNN)](https://arxiv.org/abs/1606.04671)
  - [Piggyback (PB)](https://arxiv.org/abs/1801.06519)
  - [Learning to Prompt (L2P)](https://arxiv.org/abs/2112.08654) (task-aware variant)

- **Regularization-based**
  - [Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796) (online variant)
  - [Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282) (online variant)
  - [Memory Aware Synapses (MAS)](https://arxiv.org/abs/1711.09601)

---------------------------------------------------------------------------------------------------------

## ⚡ Dataset [[download](https://zenodo.org/record/8065754)]

The dataset used for the CL-MASR benchmark is extracted from [Common Voice 13](https://commonvoice.mozilla.org/en/datasets) (see [reference paper](https://arxiv.org/abs/1912.06670)).
Each of the 20 languages in the dataset includes approximately 10 hours of training material, with an additional 1 hour designated for validation and another 1 hour for testing purposes.

Download the dataset from [here](https://zenodo.org/record/8065754) and extract it to a data folder of your choice (`CL-MASR` by default).

---------------------------------------------------------------------------------------------------------

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

2. Navigate to `<path-to-repository>/benchmarks/CL_MASR` in your file system, open a terminal, and run the following commands:

   ```shell
   pip install -r ../../requirements.txt    # Install base dependencies
   pip install -r extra-requirements.txt    # Install additional dependencies
   ```

   These commands will install the necessary dependencies for the benchmark, including both the base requirements and the additional requirements.

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

### Running an experiment

Navigate to `<path-to-repository>/benchmarks/CL_MASR/<model>`, open a terminal and run:

```bash
python train_<cl-method>.py hparams/train_<cl-method>.yaml --data_folder <path-to-data-folder>
```

**NOTE**: to profile the model (optional), install `ptflops` and `torchinfo` as additional dependencies.

**NOTE**: multi-GPU training is currently not supported.

### Analyzing the results

Navigate to `<path-to-repository>/benchmarks/CL_MASR`, open a terminal and run:

```bash
python analyze_logs.py <path-to-folder-containing-model-logs>
```

This command will recursively retrieve and analyze all log files that are named according to the
format `<cl-method>_base=<comma-separated-base-locales>_new=<comma-separated-new-locales>.txt`
(this is the default naming convention followed in all the training scripts).
You can find the resulting performance metric summaries and plots in `<path-to-folder-containing-model-logs>`.
See the help (`python analyze_logs.py -h`) for advanced configuration options.

**NOTE**: make sure to specify the `--im_refs` and `--fwt_refs` arguments that correspond to the given model (default to Whisper large-v2).

**NOTE**: to plot the results (optional), install `matplotlib` and/or `plotly` as additional dependencies.

---------------------------------------------------------------------------------------------------------

## 📈️ Results

| Release  |         Hyperparameters         | Average AWER | Average BWT | Average IM | Average FWT |                                       Logs                                        |     GPUs    |
|:--------:|:-------------------------------:|:------------:|:-----------:|:----------:|:-----------:|:---------------------------------------------------------------------------------:|:-----------:|
| 07-06-23 |  whisper/hparams/train_ft.yaml  |    98.50     |   -84.58    |   -4.16    |    -0.83    | [Link](https://www.dropbox.com/sh/qfcookgmnacz00w/AAA4kTKoeybMV1ikJLZgHjPma?dl=0) | 1xV100 32GB |
| 07-06-23 |  whisper/hparams/train_er.yaml  |    50.83     |   -13.20    |   -0.81    |    -4.17    | [Link](https://www.dropbox.com/sh/oer2tc5m2542gg4/AAAR9MVlirpVSq_Yte6BvtJTa?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_agem.yaml |    81.08     |   -55.85    |    0.20    |    -5.19    | [Link](https://www.dropbox.com/sh/0a9vbeo4e22s5rd/AAC_KUVIbl6i4MS7Qc6wu-Qva?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_pnn.yaml  |    44.12     |     0.00    |    3.18    |    -8.16    | [Link](https://www.dropbox.com/sh/75nnsef8ssz8g8d/AADrL5OKh8Kilp7gfA5pPYcOa?dl=0) | 1xV100 32GB |
| 07-06-23 |  whisper/hparams/train_pb.yaml  |    43.95     |     0.00    |    3.51    |    -8.50    | [Link](https://www.dropbox.com/sh/ipimdg4f5tvu293/AACbk6OYQiOU5NiPa18lUv3ra?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_ewc.yaml  |    98.04     |   -68.32    |    2.87    |    -7.85    | [Link](https://www.dropbox.com/sh/faagbafvkdbs8t9/AAAV9ok_Z36rwzwmMQEuELvga?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_lwf.yaml  |    95.76     |   -77.50    |    0.00    |    -4.98    | [Link](https://www.dropbox.com/sh/hgaw74v233l4awb/AAAtAnBWm0IkQcmLa6mAzI_Da?dl=0) | 1xV100 32GB |
| 07-06-23 |   wavlm/hparams/train_ft.yaml   |    91.61     |   -54.67    |  -10.19    |    -0.21    | [Link](https://www.dropbox.com/sh/bqb4k3y4gmjt4l7/AACzJc60SAdDaLkESWffEV6ka?dl=0) | 1xV100 32GB |
| 07-06-23 |   wavlm/hparams/train_er.yaml   |    60.79     |    -8.96    |   -7.62    |    -2.77    | [Link](https://www.dropbox.com/sh/cainxpnpkvxjh37/AAAzGYnjoFbB1xWoIuqFzNVda?dl=0) | 1xV100 32GB |
| 07-06-23 |  wavlm/hparams/train_agem.yaml  |    72.54     |    13.59    |   35.29    |   -45.69    | [Link](https://www.dropbox.com/sh/obt9cj3cthfnfdn/AACdj-UyzmpeME6_w_CDaZe7a?dl=0) | 1xV100 32GB |
| 07-06-23 |  wavlm/hparams/train_pnn.yaml   |    66.07     |     0.00    |   12.95    |   -23.34    | [Link](https://www.dropbox.com/sh/t4nq5qx0nl8zbtn/AACi9r_AtyJqdrbxHEZ20IxGa?dl=0) | 1xV100 32GB |
| 07-06-23 |   wavlm/hparams/train_pb.yaml   |    61.87     |     0.00    |    2.75    |   -13.15    | [Link](https://www.dropbox.com/sh/gsziwbjqfwzrlyi/AACSDG6Y0fx93JNWnuzQufhia?dl=0) | 1xV100 32GB |
| 07-06-23 |  wavlm/hparams/train_ewc.yaml   |    86.98     |   -39.54    |   -4.26    |    -6.13    | [Link](https://www.dropbox.com/sh/dumg837b3n7f8pg/AABYrnDdsLIkS7DYSUbx-MGJa?dl=0) | 1xV100 32GB |
| 07-06-23 |  wavlm/hparams/train_lwf.yaml   |    87.17     |   -26.03    |   10.42    |   -20.82    | [Link](https://www.dropbox.com/sh/mm68n9ebt6484v2/AAAw3Ckrlfagi9VcJ6qBK9yxa?dl=0) | 1xV100 32GB |

Raw experiment logs are available [here](https://www.dropbox.com/sh/vfxrb7z0p79vhqt/AAC5ftGfpibhuLlyQ8D9NfWAa?dl=0).
We do not include the checkpoints due to storage limits (each experiment with Whisper large-v2 generates ~125 GB of checkpoint data).

Analyses generated via `analyze_logs.py` are available [here](https://www.dropbox.com/sh/59uoq0ys53zf2oj/AABUvEBjNt1jUnqHdZaLdMV6a?dl=0).

All the experiments were run on 5 CentOS Linux machines with an Intel(R) Xeon(R) Silver 4216 Cascade Lake CPU
with 32 cores @ 2.10 GHz, 64 GB RAM and an NVIDIA Tesla V100 SXM2 @ 32 GB with CUDA Toolkit 11.4.
With the specified hardware configuration, approximately 10 days are necessary to complete all the experiments.

**NOTE**: the checkpoint for WavLM large pretrained on the base languages is available [here](https://www.dropbox.com/sh/tbjuwi9yo1fv2ez/AABILEMY9yAhHMwhTlvyw69Pa?dl=0).

---------------------------------------------------------------------------------------------------------

## @ Citing

If you use the CL-MASR benchmark, please cite:

```bibtex
@article{dellalibera2023clmasr,
  author = {Luca Della Libera and Pooneh Mousavi and Salah Zaiem and Cem Subakan and Mirco Ravanelli},
  title = {{CL-MASR}: A Continual Learning Benchmark for Multilingual {ASR}},
  journal = {arXiv preprint arXiv:2310.16931},
  year = {2023},
  url = {https://arxiv.org/abs/2310.16931},
}
```

If you use SpeechBrain, please cite:

```bibtex
@article{ravanelli2021speechbrain,
  author = {Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  title = {{SpeechBrain}: A General-Purpose Speech Toolkit},
  journal = {arXiv preprint arXiv:2106.04624},
  year = {2021},
  url = {https://arxiv.org/abs/2106.04624},
}
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
