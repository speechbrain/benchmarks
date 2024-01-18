# SpeechBrain-MOABB: Tutorials
## Introduction
Here we provide tutorials for introducing users to SpeechBrain-MOABB.

Specifically, we provide:
* Tutorial no. 1 **Setting up EEG decoding** (`tutorials/sbm_cli-part1.ipynb`). 
In this tutorial we provide a brief introduction of SpeechBrain-MOABB and of its first command-line interface. 
This interface allows users to run a full experiment on a target decoding problem (e.g., P300 decoding on a MOABB dataset) with known hyperparameters (i.e., no hyperparameter tuning is performed).
Models are trained and evaluated adopting a multi-seed initialization strategy, for providing performance estimates robust to fluctuations due to random initialization.
* Tutorial no. 2 **Setting up hyper-parameter tuning** (`tutorials/sbm_cli-part2.ipynb`).
This tutorial covers the second command-line interface of SpeechBrain-MOABB, devoted at performing multi-step hyperparameter tuning of the hyperparameters characterizing the entire decoding pipeline. 
Notably, the tuned hyperparameters can include both data pre-processing, network architecture, network training, and data augmentation steps of the deep learning-based EEG decoding pipeline. 
As for the previous command-line interface, here models are trained and evaluated adopting a multi-seed initialization strategy, for providing robust performance estimates.
* Tutorial no. 3 **Integrating braindecode models** (`tutorials/sbm_braindecode.ipynb`).
This tutorial shows how to use braindecode models in SpeechBrain-MOABB, designing a robust and comprehensive decoding pipeline with SpeechBrain-MOABB while taking advantage from the large set of models supported in braindecode.

Moreover, we also provide a tutorial for replicating the results reported in the paper associated to SpeechBrain-MOABB (`tutorials/sbm_results-paper.ipynb`), to ensure transparency of the results reported in the paper. 

## ▶️ Quickstart
All tutorials were designed to run on Google Colaboratory using GPU. 
Users can copy and paste these tutorials in Google Colaboratory to use them.

 
## 📧 Contact

For any questions or inquiries, feel free to reach Davide Borra, Ph.D., University of Bologna ([davide.borra2@unibo.it](mailto:davide.borra2@unibo.it)).

## **Citing**

If you use SpeechBrain-MOABB for your research or business, please cite:

[The link to the official paper will be available soon]

Please also cite SpeechBrain:

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
