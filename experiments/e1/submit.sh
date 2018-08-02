#!/bin/bash
# ganExp1 1

#SBATCH --array=0-1
#SBATCH --job-name=ganExp1
#SBATCH --qos=high
#SBATCH --mem=100mb
#SBATCH --partition dpart
###SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e1/logs/slurmOut.txt
###SBATCH --error err.txt

echo "Hi, I am task:"
echo ${SLURM_ARRAY_TASK_ID}

cd ~/projects/template/mlworkflow/tests/
python test2.py --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT}
