#!/bin/zsh
# Please consult the README.md file for instructions on how to run the benchmark.

hub='facebook/hubert-large-ll60k'
num_layers='25'
encoder_dim='1024'
output_folder='/path/to/output'
declare -a DatasetsFolders= ('path/to/LibriSpeech' 'path/to/IEMOCAP')
declare -a ConsideredTasks=('LibriSpeechASR' 'IEMOCAP')
declare -a DownStreams=('BiLSTM' 'ecapa_tdnn')
for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder =${DatasetsFolders[i]}
	python $task/$downstream/train.py $task/$downstream/hparams/ssl.yaml --num_layers_ssl $num_layers --ssl_hub $hub --encoder_dim $encoder_dim --output_folder $output_folder/$task/$downstream --data_folder $dataset_folder
done
