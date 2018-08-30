#!/bin/bash
# Name: birds5
# ID: 31

###SBATCH --array=0-9
#SBATCH --job-name=birds5
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output /fs/vulcan-scratch/krusinga/projects/ganProbability/experiments/e31/logs/out.out
###SBATCH --error err.txt

echo "Hi, I am task:"

cd /fs/vulcan-scratch/krusinga/projects/ganProbability/
module load cuda
python run.py --stage 2 --id 31 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml
