#!/bin/bash

###########################################################
# Script to run downstream evaluation training, optionally with multiple seeds.
# This script loops over seeds and trains different models.
# At the end, the final performance is computed with the aggregate_results.py script that provides the average performance.
#
# Usage:
# ./run_experiments.sh --hparams benchmarks/DASB/LibriSpeech/ASR/hparams/LSTM/train.yaml --data_folder LibriSpeech --cached_data_folder cache/ \
# --output_folder results/LibriSpeech/ASR/encodec/LSTM --task ASR --dataset LibriSpeech --seed 1986 --nruns 2 --eval_metric WER  --tokens_folder LibriSpeech/extraction-emb/speech_tokenizer/save/librispeech/

#
# Authors:
# - Pooneh Mousavi (2024)
###########################################################

# Initialize variables
hparams=""
data_folder=""
cached_data_folder=""
output_folder=""
task=""
dataset=""
seed=""
nruns=""
eval_metric="acc"
eval_set="test"
rnd_dir=False
additional_flags=""


# Function to print argument descriptions and exit
print_argument_descriptions() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --hparams hparams_path            Hparam YAML file"
    echo "  --data_folder data_folder_path    Data folder path"
    echo "  --cached_data_folder cache_path   Cached data folder path"
    echo "  --output_folder output_path       Output folder path"
    echo "  --task task                       downstream task"
    echo "  --dataset dataset                 dataset"
    echo "  --seed random_seed                Seed (random if not specified)"
    echo "  --nruns num_runs                  Number of runs"
    echo "  --eval_metric metric              Evaluation metric (e.g., acc or WER)"
    echo "  --eval_set dev or test            Evaluation set. Default: test"
    echo "  --rnd_dir                         If True the results are stored in a subdir of the output folder with a random name (useful to store all the results of an hparam tuning).  Default: False"
    exit 1
}


# Parse command line
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --hparams)
      hparams="$2"
      shift
      shift
      ;;

    --data_folder)
      data_folder="$2"
      shift
      shift
      ;;

    --cached_data_folder)
      cached_data_folder="$2"
      shift
      shift
      ;;

    --output_folder)
      output_folder="$2"
      shift
      shift
      ;;

      --task)
      task="$2"
      shift
      shift
      ;;

      --dataset)
      dataset="$2"
      shift
      shift
      ;;

    --seed)
      seed="$2"
      shift
      shift
      ;;

    --nruns)
      nruns="$2"
      shift
      shift
      ;;

    --eval_metric)
      eval_metric="$2"
      shift
      shift
      ;;

    --eval_set)
      eval_set="$2"
      shift
      shift
      ;;

    --rnd_dir)
      rnd_dir="$2"
      shift
      shift
      ;;


    --help)
      print_argument_descriptions
      ;;

    -*|--*)
      additional_flags+="$1 $2 " # store additional flags
      shift # past argument
      ;;


    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done


# Check for required arguments
if  [ -z "$hparams" ] ||[ -z "$data_folder" ] || [ -z "$output_folder" ]  || [ -z "$nruns" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
fi

# Manage Seed (optional argument)
seed="${seed:-$RANDOM}"


if [ "$rnd_dir" = True ]; then
    if [[ ! -z "$ORION_TRIAL_ID" ]]; then
      # Use the Orion Trial ID to ensure interrupted trials are resumed
      output_folder="$output_folder/$ORION_TRIAL_ID"
    else
      rnd_dirname=$(tr -dc 'a-zA-Z' < /dev/urandom | head -c 6)
      output_folder="$output_folder/$rnd_dirname"
    fi
fi

# Make sure  the output_folder is created
mkdir -p $output_folder

# Print command line arguments and save to file
{
    echo "hparams: $hparams"
    echo "data_folder: $data_folder"
    echo "cached_data_folder: $cached_data_folder"
    echo "output_folder: $output_folder"
    echo "task: $task"
    echo "dataset: $dataset"
    echo "seed: $seed"
    echo "nruns: $nruns"
    echo "eval_metric: $eval_metric"
    echo "eval_set: $eval_set"
    echo "rnd_dir: $rnd_dir"
    echo "additional flags: $additional_flags"
} | tee "$output_folder/flags.txt"


# Creating output folder
mkdir -p $output_folder
mkdir -p $data_folder
mkdir -p $cached_data_folder

# Function to run the training experiment
run_experiment() {

eval python $dataset/$task/train.py $hparams  --cached_data_folder=$cached_data_folder --seed=$seed --data_folder=$data_folder --output_folder=$output_folder_exp \
$additional_flags

}

# Run multiple training experiments (with different seeds)
for i in $(seq 0 1 $(( nruns - 1 ))); do
  ((run_idx = i + 1))
  run_name=run"$run_idx"
  output_folder_exp="$output_folder"/"$run_name"/$seed

  run_experiment  $output_folder_exp


  # Changing Random seed
  seed=$((seed+1))
done


echo 'Final Results (Performance Aggregation)'
python utils/aggregate_results.py $output_folder "$eval_metric" | tee -a  $output_folder/aggregated_performance.txt
