#!/bin/bash
# Name: {0} 
# ID: {1}

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name={0}
#SBATCH --qos=default
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output {2}out_0.txt
###SBATCH --error err.txt

echo "Hi, I am task:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 1