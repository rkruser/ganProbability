#!/bin/bash
# Name: birdsnapInfogan 
# ID: 26

#SBATCH --array=0-9
#SBATCH --job-name=birdsnapInfogan
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output /cfarhomes/krusinga/ganProb/ganProbProject/experiments/e26/logs/out_%a.txt
###SBATCH --error err.txt

echo "Hi, I am task:"
echo ${SLURM_ARRAY_TASK_ID}

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT} --stage 2
