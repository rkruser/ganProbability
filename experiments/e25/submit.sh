#!/bin/bash
# Name: birdsnapInfogan 
# ID: 25

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name=birdsnapInfogan
###SBATCH --qos=default
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /cfarhomes/krusinga/ganProb/ganProbProject/experiments/e25/logs/out_0.txt
###SBATCH --error err.txt

echo "Hi, I am task:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 1 --id 25 --masterconfig /cfarhomes/krusinga/ganProb/ganProbProject/master.yaml
