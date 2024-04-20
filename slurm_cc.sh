#!/bin/bash
#SBATCH --time=329:55:00
#SBATCH --array=1-72%30     # Number of parameter sets
#SBATCH --mem=40000
#SBATCH --partition=ultralong
#SBATCH --job-name=ccfraud
#SBATCH --qos=normal
#SBATCH -o ./log/cc.%A.%a.out # STDOUT
#SBATCH -e ./log/cc.%A.%a.err # STD error

# Load necessary modules or set up environment if needed
# module purge
module load python/3.11.7-gcc114-base

# Run the Python script with the corresponding parameter from the CSV file
export PYTHONUNBUFFERED=TRUE
python3 job_ccfraud.py $SLURM_ARRAY_TASK_ID