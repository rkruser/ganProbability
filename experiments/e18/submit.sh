#!/bin/bash
# Name: mnistStage1 
# ID: 18

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name=mnistStage1
#SBATCH --qos=default
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e18/logs/out_0.txt
###SBATCH --error err.txt

echo "Hi, I am task:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 1
