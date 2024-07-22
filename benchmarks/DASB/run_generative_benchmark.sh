#!/bin/bash
# Please consult the README.md file for instructions on how to run the benchmark.

tokenizer_name=$1
output_folder='/path/to/output'
declare -a DatasetsFolders=('path/to/Libri2Mix' 'path/to/VoiceBank')
declare -a ConsideredTasks=('Libri2Mix/separation' 'VoiceBank/enhancement')
declare -a DownStreams=('conformer' 'conformer')
for i in "${!ConsideredTasks[@]}"; do
        task=${ConsideredTasks[i]}
        downstream=${DownStreams[i]}
        dataset_folder=${DatasetsFolders[i]}
        python $task/$downstream/train_$tokenizer_name.py $task/$downstream/hparams/train_$tokenizer_name.yaml   --output_folder $output_folder/$tokenizer_name/$task/$downstream --data_folder $dataset_folder 
done
