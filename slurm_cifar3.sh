#!/bin/bash
#SBATCH --time=329:55:00
#SBATCH --array=2001-3000%50     # Number of parameter sets
#SBATCH --mem=40000
#SBATCH --partition=ultralong
#SBATCH --job-name=cifar10
#SBATCH --qos=normal
#SBATCH -o ./log/cifar10.%A.%a.out # STDOUT
#SBATCH -e ./log/cifar10.%A.%a.err # STD error

# Load necessary modules or set up environment if needed
# module purge
module load python/3.11.7-gcc114-base

# Run the Python script with the corresponding parameter from the CSV file
export PYTHONUNBUFFERED=TRUE
python3 job_cifar10.py $SLURM_ARRAY_TASK_ID

