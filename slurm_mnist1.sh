#!/bin/bash
#SBATCH --time=329:55:00
#SBATCH --array=1-1000%500     # Number of parameter sets
#SBATCH --mem=40000
#SBATCH --partition=ultralong
#SBATCH --job-name=mnist
#SBATCH --qos=normal
#SBATCH -o ./log/mnist.%A.%a.out # STDOUT
#SBATCH -e ./log/mnist.%A.%a.err # STD error

# Load necessary modules or set up environment if needed
# module purge
module load python/3.11.7-gcc114-base

# Run the Python script with the corresponding parameter from the CSV file
export PYTHONUNBUFFERED=TRUE
python3 job_mnist.py $SLURM_ARRAY_TASK_ID

