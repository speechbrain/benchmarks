#!/bin/bash

###########################################################
# Script to run leave-one-subject-out and/or leave-one-session-out training, optionally with multiple seeds.
# This script loops over the different subjects and sessions and trains different models.
# At the end, the final performance is computed with the aggregate_results.py script that provides the average performance.
#
# Usage:
# ./run_experiments.sh --hparams=hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data \
# --output_folder=results/MotorImagery/BNCI2014001/EEGNet --nsbj=9 --nsess=2 --seed=1986 --nruns=2 --number_of_epochs=10
#
# Authors:
# - Pooneh Mousavi (2024)
###########################################################

# Initialize variables
data_folder=""
cached_data_folder=""
output_folder=""
task=""
downstream=""
tokenizer_name=""
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
    echo "  --data_folder data_folder_path    Data folder path"
    echo "  --output_folder output_path       Output folder path"
    echo "  --task task                       downstream task"
    echo "  --downstream downstream           probing head"
    echo "  --tokenizer_name tokenizer_name   tokenizer choice"
    echo "  --dataset dataset               dataset"
    echo "  --seed random_seed                Seed (random if not specified)"
    echo "  --nruns num_runs                  Number of runs"
    echo "  --eval_metric metric              Evaluation metric (e.g., acc or f1)"
    echo "  --eval_set dev or test            Evaluation set. Default: test"
    echo "  --rnd_dir                         If True the results are stored in a subdir of the output folder with a random name (useful to store all the results of an hparam tuning).  Default: False"
    exit 1
}


# Parse command line
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --data_folder)
      data_folder="$2"
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
    
    
    --downstream)
      downstream="$2"
      shift
      shift
      ;;   

      --tokenizer_name)
      tokenizer_name="$2"
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
if  [ -z "$data_folder" ] || [ -z "$output_folder" ]  || [ -z "$nruns" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
fi

# Process eval_set argument
if [ "$eval_set" = "dev" ]; then
  metric_file=valid_metrics.pkl
elif [ "$eval_set" = "test" ]; then
  metric_file=test_metrics.pkl
else
  echo "Invalid eval_set value: $eval_set. It can be test or dev only."
  exit 1
fi

# Manage Seed (optional argument)
seed="${seed:-$RANDOM}"



if [ "$rnd_dir" = True ]; then
    rnd_dirname=$(tr -dc 'a-zA-Z' < /dev/urandom | head -c 6)
    output_folder="$output_folder/$rnd_dirname"
fi

# Make sure  the output_folder is created
mkdir -p $output_folder

# Print command line arguments and save to file
{
    echo "hparams: $hparams"
    echo "data_folder: $data_folder"
    echo "output_folder: $output_folder"
    echo "task: $task"
    echo "downstream: $downstream"
    echo "tokenizer_name: $tokenizer_name"
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

python $dataset/$task/train.py $dataset/$task/hparams/$downstream/$tokenizer_name.yaml  --seed=$seed --data_folder=$data_folder --output_folder=$output_folder_exp\
$additional_flags --debug

}

# Run multiple training experiments (with different seeds)
for i in $(seq 0 1 $(( nruns - 1 ))); do
  ((run_idx = i + 1))
  run_name=run"$run_idx"
  output_folder_exp="$output_folder"/"$run_name"/$seed

  run_experiment  $output_folder_exp


  # Store the results
  # python utils/parse_results.py $output_folder_exp $metric_file $eval_metric | tee -a  $output_folder/$run_name\_results.txt

  # Changing Random seed
  seed=$((seed+1))
done


echo 'Final Results (Performance Aggregation)'
python utils/aggregate_results.py $output_folder $eval_metric | tee -a  $output_folder/aggregated_performance.txt
