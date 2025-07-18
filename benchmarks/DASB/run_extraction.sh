#!/bin/bash

###########################################################
# Script to extracts and save tokens from dataset.
#
# Usage:
# ./ $run_extraction.sh  --data_folder LibriSpeech --output_folder results/LibriSpeech/ASR/encodec/LSTM --tokenizer encodec --dataset LibriSpeech

# Authors:
# - Pooneh Mousavi (2024)
###########################################################

# Initialize variables
data_folder=""
output_folder=""
tokenizer=""
dataset=""
save_embedding=False
additional_flags=""


# Function to print argument descriptions and exit
print_argument_descriptions() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --data_folder data_folder_path    Data folder path"
    echo "  --output_folder output_path       Output folder path"
    echo "  --tokenizer tokenizer             tokenizer"
    echo "  --dataset dataset                 dataset"
    echo "  --save_embedding save_embedding   If True the the embedding are saved. Default: False"
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

    --tokenizer)
      tokenizer="$2"
      shift
      shift
      ;;

    --dataset)
      dataset="$2"
      shift
      shift
      ;;

    --save_embedding)
      save_embedding="$2"
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
if  [ -z "$tokenizer" ] ||[ -z "$data_folder" ] || [ -z "$output_folder" ]  || [ -z "$dataset" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
fi


# Make sure  the output_folder is created
mkdir -p $output_folder

# Print command line arguments and save to file
{
    echo "data_folder: $data_folder"
    echo "output_folder: $output_folder"
    echo "tokenizer: $tokenizer"
    echo "dataset: $dataset"
    echo "save_embedding: $save_embedding"
    echo "additional flags: $additional_flags"
} | tee "$output_folder/flags.txt"


# Creating output folder
mkdir -p $output_folder
mkdir -p $data_folder

python $dataset/extraction/extract.py $dataset/extraction/hparams/$tokenizer.yaml --data_folder=$data_folder --output_folder=$output_folder --save_embedding=$save_embedding \
$additional_flags
