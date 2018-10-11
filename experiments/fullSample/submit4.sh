#!/bin/bash
# Name: {0}
# ID: {1}

#SBATCH --array=0-9
#SBATCH --job-name={0}
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output {2}out_%a.out
###SBATCH --error err.txt

echo "Hello:"
echo ${{SLURM_ARRAY_TASK_ID}}

cd {3}
module load cuda
python run.py --stage 4 --id {1} --masterconfig {3}master.yaml  --pid ${{SLURM_ARRAY_TASK_ID}} --nprocs ${{SLURM_ARRAY_TASK_COUNT}}
