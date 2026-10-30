#!/bin/bash

# Loop through all subdirectories in the current directory
for dir in */; do
    # Remove trailing slash
    dir=${dir%/}
    # Run your Python command with the subdirectory path
    python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/plot_dist_thesis.py -json_path "$dir/results.json"
done
