#!/bin/bash
# Name: mnistInfogan 
# ID: 32

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name=mnistInfogan
###SBATCH --qos=default
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /cfarhomes/krusinga/ganProb/ganProbProject/experiments/e32/logs/out_0.txt
###SBATCH --error err.txt

echo "Hi, I am task:"

cd /cfarhomes/krusinga/ganProb/ganProbProject
module load cuda
python run.py --stage 1 --id 32 --masterconfig /cfarhomes/krusinga/ganProb/ganProbProject/master.yaml
