#!/bin/bash
#SBATCH -n 1                        # Use 1 CPU
#SBATCH -t 02:00:00                 # Set a 2-hour time limit (adjust if needed)
#SBATCH --partition=use-everything  # Use available nodes
#SBATCH --output=logs/transfer_%j.out  # Log file
#SBATCH --error=logs/transfer_%j.err   # Error log

# Define source and destination
SRC="/om2/user/sahil003/myenv/activation_dump/"
DEST="/om2/scratch/sahil003/activation_dump/"

echo "Starting transfer from $SRC to $DEST on $(hostname)"

# Use rsync to copy files efficiently
rsync -av --progress "$SRC" "$DEST"

echo "Transfer complete."
