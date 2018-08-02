#!/bin/bash
# Name: mnist_distribution 
# ID: 14

#SBATCH --array=0-9
#SBATCH --job-name=mnist_distribution
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e14/logs/out_%a.out
###SBATCH --error err.txt

echo "Hi, I am task:"
echo ${SLURM_ARRAY_TASK_ID}

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT} --stage 2
