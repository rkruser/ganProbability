#!/bin/bash
# Name: stage1 
# ID: 58

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name=stage1
###SBATCH --qos=default
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e58/logs/out.out
###SBATCH --error err.txt

echo "Hello:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 1 --id 58 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml
