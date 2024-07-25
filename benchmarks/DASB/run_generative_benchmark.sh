#!/bin/bash
# Please consult the README.md file for instructions on how to run the benchmark.

tokenizer_name=$1
if [[ "$tokenizer_name" == "" ]]; then
        echo "Usage: run_generative_benchmark.sh <tokenizer_name>"
        exit 1
fi

output_folder='path/to/output'
librimix_path='path/to/Libri2Mix'
voicebank_path='path/to/VoiceBank'
ljspeech_path='path/to/ljspeech'
utmos_path='path/to/utmos'
tts_args="--token_list_file_text %recipe_root%/hparams/char_en.txt --utmos_model_path $utmos_path"

declare -a DatasetsFolders=(\
        "$librimix_path" \
        "$voicebank_path" \
        "$ljspeech_path" \
        "$ljspeech_path" \
)
declare -a ConsideredTasks=(\
        'Libri2Mix/separation' \
        'VoiceBank/enhancement' \
        'LJSpeech/TTS' \
        'LJSpeech/TTS' \
)
declare -a DownStreams=(\
        'conformer' \
        'conformer' \
        'tokotron' \
        'tokotron' \
)
declare -a ExtraArgs=(\
        '' \
        '' \
        "$tts_args" \
        "$tts_args --enc_num_layers 3 --dec_num_layers 6" \
)

declare -a OutputSuffix=(\
        '' \
        '' \
        '' \
        '-small'
)

shift
script_args="$@"

for i in "${!ConsideredTasks[@]}"; do
        task=${ConsideredTasks[i]}
        downstream=${DownStreams[i]}
        dataset_folder=${DatasetsFolders[i]}
        extra_args=${ExtraArgs[i]}
        suffix=${OutputSuffix[i]}
        recipe_root="$task/$downstream"
        recipe_extra_args="$script_args ${extra_args//%recipe_root%/$recipe_root}"
        set -- "$recipe_extra_args"
        echo "${tokenizer_name}/${task}/${downstream}"
        python $task/$downstream/train_$tokenizer_name.py \
                $task/$downstream/hparams/train_$tokenizer_name.yaml  \
                --output_folder $output_folder/$tokenizer_name/$task/$downstream$suffix \
                --data_folder $dataset_folder \
                $@
done
