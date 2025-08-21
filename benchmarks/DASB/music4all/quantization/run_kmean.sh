#!/bin/bash

# Ensure the script is executable with: chmod +x run_experiments.sh

# Number of iterations
iterations=24

# Base command to run the experiment
base_command="python train.py hparams/train_discrete_ssl.yaml --data_folder /home/ubuntu/music4all --data_cache_folder /home/ubuntu/music4all --n_clusters 1000"

# Loop over the iterations
for ((i=0; i<iterations; i++))
do
    echo "Running experiment with layer_id $i..."

    # Generate the experiment name
    experiment_name="mert_K1000_L$i"
    
    # Construct the full command with the current layer_id and experiment_name
    experiment_command="$base_command --layer_id $i --experiment_name $experiment_name"
    
    # Run the experiment
    $experiment_command
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Experiment failed at layer_id $i. Exiting..."
        exit 1
    fi

    echo "Completed experiment with layer_id $i."
done

echo "All $iterations experiments completed successfully."
