#!/bin/bash
# {0} {1}

#SBATCH --array=0-1
#SBATCH --job-name={0}
#SBATCH --qos=high
#SBATCH --mem=100mb
#SBATCH --partition dpart
###SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output {2}/out.txt
###SBATCH --error err.txt

echo "Hi, I am task:"
echo ${{SLURM_ARRAY_TASK_ID}}

cd ~/projects/template/mlworkflow/tests/
python test2.py --pid ${{SLURM_ARRAY_TASK_ID}} --nprocs ${{SLURM_ARRAY_TASK_COUNT}}
