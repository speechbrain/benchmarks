# Benchmark for speech self-supervised representations
This recipe includes scripts to evaluate self-supervised represenations on a list of downstreams tasks that includes ASR, speaker verification, emotion recognition and intent classification. The main particularity of this benchmark is that it allows the user to select the probing head wanted for downstream training. It has been shown that model's performances depend heavily on this choice. More details about the motivation, the tasks and the results can be found in the following paper : 

```bibtex
@article{zaiem2023speech,
  title={Speech Self-Supervised Representation Benchmarking: Are We Doing it Right?},
  author={Zaiem, Salah and Kemiche, Youcef and Parcollet, Titouan and Essid, Slim and Ravanelli, Mirco},
  journal={arXiv preprint arXiv:2306.00452},
  year={2023}
}
```
## 🛠️️ Installation
The extra requirements are only needed if you want to use the n-gram language model during the decoding for English ASR experiments. If it is the case, navigate to `<path-to-repository>/benchmarks/SSL_BENCHMARK` and run : 

```bash
pip install -r extra_requirements.txt
```


## ▶️ Quickstart

To run a downstream evaluation for a given SSL model on huggingface you will need to  :
* Change the SSL related values in the run\_benchmark.sh file, specifying the HF hub, the encoder dimension (size of every frame vector), and the number of layers.
* Choose a set of tasks among the ones listed  and for every task a downstream architecture among the existing ones.
* Change the variable defined in run\_benchmark.sh with two lists of equal sized where to every task  in "ConsideredTasks" corresponds in the same index in "Downstreams" the downstream architecture.
* If you want to run two downstream decoders on the same task, just put it twice in the first list with different corresponding decoders below.

then, run : 

```
bash run_benchmark.sh
```
The current bash file provides an example with HuBERT Large as the SSL encoder and LibriSpeechASR and IEMOCAP as downstream tasks, respectively with BiLSTM and ECAPA-TDNN probing heads. It can be run by changing the data folder to your local setting folders containing LS and IEMOCAP. 

If you want to run only one experiment on a considered task, let us take the example of LibriSpeech using the huggingface model link : mygroup/mySSLModel, that outputs frame vectors of dimension 768 and has 13 internal layers, than you should run : 

```
python LibriSpeech/LSTM/train.py LibriSpeech/LSTM/hparams/ssl.yaml --ssl_hub mygroup/mySSLModel --encoder_dim 768 --num_layers_ssl 13 --output_folder my-output-folder --data_folder mypath/to/LibriSpeech 
``` 

## Tasks and downstream heads

| Task                                     | First Probing Head    | Second Probing Head | Dataset Download                                         |
|------------------------------------------|-----------------------|---------------------|----------------------------------------------------------|
| LibriSpeech ASR                          | BiLSTM                | Conformer           | https://www.openslr.org/12                               |
| Buckeye ASR                              | BiLSTM                | ContextNet          | https://buckeyecorpus.osu.edu/                           |
| CommonVoice Low-Resource  ASR            | BiLSTM                | Linear              | https://commonvoice.mozilla.org/fr                       |
| VoxCeleb1 Automatic Speaker Verification | X-Vectors             | ECAPA-TDNN          | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html |
| IEMOCAP Emotion Recognition              | Time-Pooling + Linear | ECAPA-TDNN          | https://sail.usc.edu/iemocap/                            |
| SLURP Intent Classification              | Time-Pooling + Linear | BiLSTM + Linear     | https://zenodo.org/record/4274930                        |

The n-gram-based language model used during decoding for the English ASR experiments can be downloaded here : https://www.openslr.org/resources/11/4-gram.arpa.gz. After download, in order to use it, you need to specify its path in the ngram\_lm\_path variable, and activate language modelling during decoding by passing language\_modelling True. 

## 📈️ Results

The following table shows results obtained with 2 different SSL encoders, and with two different probing heads for every task. The full table can be found in the paper presenting the benchmark, cited above. 

| Models/Tasks                    | SSL Params | LibriSpeech Train-clean-100 | Buckeye ASR |  Welsh | Basque  |  VoxCeleb1 |       IEMOCAP      | SLURP Scenario identification |
|---------------------------------|:----------:|:---------------------------:|:-----------:|:------:|:-------:|:----------:|:------------------:|:-----------------------------:|
| Evaluation Metrics              |            |             WER             |     WER     |   WER  |   WER   |     EER    |      Accuracy      |            Accuracy           |
| First Downstream Architectures  |            |            BiLSTM           |    BiLSTM   | BiLSTM |  BiLSTM |  X-Vectors | Time-pool + Linear |       Time-pool + Linear      |
| Data2vec Base                   |    93.8M   |             5.30            |    37.26    |  54.00 |  46.37  |    5.43    |        63.0        |              56.9             |
| Data2vec Large                  |   314.3M   |             3.10            |    22.63    |  44.32 |  38.23  |    4.89    |        64.1        |              69.8             |
| Second Downstream Architectures |            |          Conformer          |  ContextNet |   Lin  |   Lin   | ECAPA-TDNN |     ECAPA-TDNN     |        BiLSTM + Linear        |
| Data2vec Base                   |    93.8M   |             5.85            |    40.53    |  77.49 |  75.26  |    3.75    |         72         |              73.4             |
| Data2vec Large                  |   314.3M   |             3.43            |    25.26    |  69.09 |  63.31  |    2.67    |        71.3        |              79.9             |

All the experiments presented in this work, including these with large SSL encoders such as Data2Vec Large should be able to run on a 32Gb V100 GPU with the batch sizes provided in the yaml files. These batch sizes may be increased for faster training with smaller "Base" SSL encoders. The next table provides estimates of training time for every couple (task, probing head) present in the paper, using Data2vec Large as the SSL encoder. For more details, we provide also the logs (and checkpoints!) of experiments done with Data2Vec Base and Large versions :

|                   Task                   | First Probing Head - Training Duration | Second Probing Head - Training Duration |                            Data2Vec Base Logs                            | Data2Vec Large Logs                                                      |
|:----------------------------------------:|:--------------------------------------:|:---------------------------------------:|:------------------------------------------------------------------------:|--------------------------------------------------------------------------|
| LibriSpeech ASR                          | 12 hours                               | 23 hours                                | https://drive.google.com/drive/folders/1qOPyNVbxXst6XOAogcU5FQLR5O2p8SzI | https://drive.google.com/drive/folders/1pNMDsjZDSXMSSp3N0aV9SCja4rPlFAUJ |
| Buckeye ASR                              | 6 hours                                | 7 hours                                 | https://drive.google.com/drive/folders/1wr\_DRXM3MkSZLzhSygeuzGfwP2Ek63bc | https://drive.google.com/drive/folders/13XechHlu4Kj4gg-E7vfHfvE0qG1GSItj |
| CommonVoice Low-Resource ASR             | 3h30                                   | 2 hours                                 | https://drive.google.com/drive/folders/1PaRWMTPe0hltTgCEfTptHc4iOwQMgZ8o | https://drive.google.com/drive/folders/1OVzqBE-3tRYDJ9Wi8Zf4q6-RI82lCX5D |
| VoxCeleb1 Automatic Speaker Verification | 23 hours                               | 25 hours                                | https://drive.google.com/drive/folders/1Jt9obJZL-c1VwCuhu5i7ul2zreZ7hqru | https://drive.google.com/drive/folders/19Y1vOzmUEH\_40HPkSb2nlUEKdU6nataE |
| IEMOCAP Emotion Recognition              | 1h30                                   | 2 hours                                 | https://drive.google.com/drive/folders/15RBDGU7r7rIH3QIDW4ZvknstQyOVb\_eJ | https://drive.google.com/drive/folders/1F\_fU-q994Y-N6Ix05BloZi1cggoYO1BU |
| SLURP Intent Classification              | 8 hours                                | 9 hours                                 | https://drive.google.com/drive/folders/1lPRZrfyIxREq7Ol54jqF3yJtdlGaY75g | https://drive.google.com/drive/folders/1w3ICpWjmnGbyVqEdqNpta7tBqkJRzMs_ | 

## 📧 Contact

[zaiemsalah@gmail.com](mailto:zaiemsalah@gmail.com)
  
# **Citing**
If you use this benchmark, please cite :

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

