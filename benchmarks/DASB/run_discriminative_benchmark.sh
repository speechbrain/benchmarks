#!/bin/bash
# Please consult the README.md file for instructions on how to run the benchmark.

tokenizer_name=$1
if [[ "$tokenizer_name" == "" ]]; then
        echo "Usage: run_generative_benchmark.sh <tokenizer_name>"
        exit 1
fi

output_folder='/path/to/output'
declare -a DatasetsFolders=('path/to/LibriSpeech' 'path/to/CommonVoice' 'path/to/IEMOCAP' 'path/to/SLURP' 'path/to/Google-speech-commands' 'path/to/VoiceCeleb1')
declare -a ConsideredTasks=('LibriSpeech/ASR' 'CommonVoice/ASR' 'IEMOCAP/emotion_recognition' 'SLURP/intent_classification' 'Google-speech-commands/keyword-spotting' 'VoiceCeleb1/speaker_ver')
declare -a DownStreams=('LSTM' 'LSTM' 'ecapa_tdnn' 'LSTM_linear' 'Xvector','Xvector')
declare -a Locales=('cy' 'eu')
declare -a LocalesVobSize=(100 200)

shift
script_args="$@"

for i in "${!ConsideredTasks[@]}"; do
        task=${ConsideredTasks[i]}
        downstream=${DownStreams[i]}
        dataset_folder=${DatasetsFolders[i]}
        recipe_extra_args="$script_args"
        set -- "$recipe_extra_args"
        if [[ "$task" == "CommonVoice/ASR" ]]; then
                echo "${tokenizer_name}/${task}/${downstream}"
                for j in "${!Locales[@]}"; do
                        locale=${Locales[j]}
                        vocab=${LocalesVobSize[j]}
                        python $task/$downstream/train_$tokenizer_name.py $task/$downstream/hparams/train_$tokenizer_name.yaml   --output_folder $output_folder/$tokenizer_name/$task/$downstream/$locale --data_folder $dataset_folder/$locale --language $locale  --output_neurons $vocab $@
                done
        else
                python $task/$downstream/train_$tokenizer_name.py $task/$downstream/hparams/train_$tokenizer_name.yaml   --output_folder $output_folder/$tokenizer_name/$task/$downstream --data_folder $dataset_folder $@
        fi
done
