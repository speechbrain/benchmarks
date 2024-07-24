#!/bin/bash
# Please consult the README.md file for instructions on how to run the benchmark.

tokenizer_name=$1
output_folder='/path/to/output'
declare -a DatasetsFolders=('path/to/LibriSpeech' 'path/to/CommonVoice' 'path/to/IEMOCAP' 'path/to/SLURP' 'path/to/Google-speech-commands' 'path/to/VoiceCeleb1')
declare -a ConsideredTasks=('LibriSpeech/ASR' 'CommonVoice/ASR' 'IEMOCAP/emotion_recognition' 'SLURP/intent_classification' 'Google-speech-commands/keyword-spotting' 'VoiceCeleb1/speaker_ver')
declare -a DownStreams=('LSTM' 'LSTM' 'ecapa_tdnn' 'LSTM_linear' 'Xvector','Xvector')
declare -a Locales=('cy' 'eu')
declare -a LocalesVobSize=(100 200)

script_dir=$(pwd)
script_args="$@"

for i in "${!ConsideredTasks[@]}"; do
        task=${ConsideredTasks[i]}
        downstream=${DownStreams[i]}
        dataset_folder=${DatasetsFolders[i]}
        extra_args=${ExtraArgs[i]}
        set -- "$script_args $extra_args"
        cd $task/$downstream

        if [[ "$task" == "CommonVoice/ASR" ]]; then
                for j in "${!Locales[@]}"; do
                        locale=${Locales[j]}
                        vocab=${LocalesVobSize[j]}
                        echo "${tokenizer_name}/${task}/${downstream}/${locale}"
                        python train_$tokenizer_name.py hparams/train_$tokenizer_name.yaml   --output_folder $output_folder/$tokenizer_name/$task/$downstream/$locale --data_folder $dataset_folder/$locale --language $locale  --output_neurons $vocab $@
                done
        else
                python train_$tokenizer_name.py hparams/train_$tokenizer_name.yaml   --output_folder $output_folder/$tokenizer_name/$task/$downstream --data_folder $dataset_folder $@
        fi
        cd $script_dir
done
