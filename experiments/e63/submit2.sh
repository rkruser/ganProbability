#!/bin/bash
# Name: stage2
# ID: 63

#SBATCH --array=0-9
#SBATCH --job-name=stage1
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e63/logs/out_%a.out
###SBATCH --error err.txt

echo "Hello:"
echo ${SLURM_ARRAY_TASK_ID}

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 2 --id 63 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml  --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT}
