#!/bin/bash
# Name: {0} 
# ID: {1}

###SBATCH --array=0-9 #Investigate
#SBATCH --job-name={0}
###SBATCH --qos=default
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
###SBATCH --partition dpart
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output {2}out.out
###SBATCH --error err.txt

echo "Hello:"

cd {3}
module load cuda
python run.py --stage 1 --id {1} --masterconfig {3}master.yaml
