#!/bin/bash
# Name: trainMNIST 
# ID: 78

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name=trainMNIST
###SBATCH --qos=default
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e78/logs/out.out
###SBATCH --error err.txt

echo "Hello:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 6 --id 78 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml
