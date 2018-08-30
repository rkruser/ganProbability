#!/bin/bash
# Name: birdsnapInfogan0 
# ID: 36

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name=birdsnapInfogan0
###SBATCH --qos=default
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e36/logs/out.out
###SBATCH --error err.txt

echo "Hi, I am task:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 1 --id 36 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml
