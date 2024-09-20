#!/bin/bash
#SBATCH --job-name=Unet_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=basic,gpu
#SBATCH --mem=50GB  # Increased memory
#SBATCH --time=5-12:00:00
#SBATCH --gres=shard:32
#SBATCH --output=log_unet_train.out
#SBATCH --error=log_unet_train.err

#sbatch /lisc/scratch/neurobiology/zimmer/schaar/code/github/segment-anything-2/pipeline_implementation/run_script.sh

# Redirect both stdout and stderr to the same error log file
exec > log_unet_train.err 2>&1

# Run the Python script
echo "Starting Python script..."

python /lisc/scratch/neurobiology/zimmer/schaar/code/github/segment-anything-2/pipeline_implementation/sam2-video-processing.py

# End of the script
echo "Script ended."
