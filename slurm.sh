#!/bin/bash
#SBATCH --job-name=my_job_array
#SBATCH --array=1-20%10     # Number of parameter sets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-:00:00

# Load necessary modules or set up environment if needed

# Run the Python script with the corresponding parameter from the CSV file
python job.py $SLURM_ARRAY_TASK_ID > cifar10_$SLURM_ARRAY_TASK_ID.log
