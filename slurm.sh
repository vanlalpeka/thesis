#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --array=1-20%10     # Number of parameter sets
#SBATCH --time=2-00:00:00
#SBATCH --mem=8000
#SBATCH --qos=normal

# Load necessary modules or set up environment if needed

# Run the Python script with the corresponding parameter from the CSV file
# python job.py $SLURM_ARRAY_TASK_ID > mnist_$SLURM_ARRAY_TASK_ID.log
export PYTHONUNBUFFERED=TRUE
python3 job.py $SLURM_ARRAY_TASK_ID > mnist_$SLURM_ARRAY_TASK_ID.log
