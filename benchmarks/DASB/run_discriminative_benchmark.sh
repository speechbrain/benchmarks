#!/bin/zsh
# Please consult the README.md file for instructions on how to run the benchmark.

tokenizer_name=$1
output_folder='/path/to/output'
declare -a DatasetsFolders= ('path/to/LibriSpeech' 'path/to/IEMOCAP' 'path/to/SLURP' 'path/to/Google-speech-commands' 'path/to/VoiceCeleb1')
declare -a ConsideredTasks=('LibriSpeech/ASR' 'IEMOCAP/emotion_recognition' 'SLURP/intent_classification' 'Google-speech-commands/keyword-spotting' 'VoiceCeleb1/speaker_ver')
declare -a DownStreams=('BiLSTM' 'ecapa_tdnn' 'LSTM_linear' 'Xvector','Xvector')
for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder =${DatasetsFolders[i]}
	python $task/$downstream/train_$tokenizer_name.py $task/$downstream/hparams/train_$tokenizer_name.yaml   $output_folder/$tokenizer_name/$task/$downstream --data_folder $dataset_folder
done
