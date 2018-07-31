#!/bin/bash
# Name: sampling4 Experiment Unique ID: 4

#SBATCH --array=0-9 ## Inclusive
#SBATCH --job-name=batchSample
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1 # for memory
#SBATCH --time=00:30:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e4/logs/out_%a.txt
###SBATCH --error err.txt

echo "Hi, I am task:"
echo ${SLURM_ARRAY_TASK_ID}

module load cuda
cd /fs/vulcan-scratch/krusinga/projects/ganProbability
python run.py --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT}
