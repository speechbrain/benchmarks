# Quantization

This folder contains recipes for training K-means quantizers on the music4all dataset.
The quantizer maps self-supervised representations from MERT. into discrete representations.
These discrete representations can then be used as input features for downstream tasks.

You can download LJSpeech from https://sites.google.com/view/contact4music4all.

---------------------------------------------------------------------------------------------------------

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies.
To do so, simply run the following command in your terminal:

```shell
pip install -r extra_requirements.txt
```

---------------------------------------------------------------------------------------------------------

## Running an Experiment

```shell
python train.py hparams/train_discrete_ssl.yaml --data_folder <path-to-dataset>
--n_clusters 1000 \
--layer_id 7 \
--experiment_name mert_K1000_L7
```
---------------------------------------------------------------------------------------------------------

## About SpeechBrain

- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

---------------------------------------------------------------------------------------------------------

## Citing SpeechBrain

Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@article{speechbrainV1,
  author  = {Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca {Della Libera} and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Ha Nguyen and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Ga{{\"e}}lle Laperri{{\`e}}re and Mickael Rouvier and Renato De Mori and Yannick Est{{\`e}}ve},
  title   = {Open-Source Conversational {AI} with {SpeechBrain} 1.0},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {333},
  pages   = {1--11},
  url     = {http://jmlr.org/papers/v25/24-0991.html}
}
```

```bibtex
@article{ravanelli2021speechbrain,
  author  = {Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  title   = {{SpeechBrain}: A General-Purpose Speech Toolkit},
  journal = {arXiv preprint arXiv:2106.04624},
  year    = {2021},
  url     = {https://arxiv.org/abs/2106.04624},
}
```
