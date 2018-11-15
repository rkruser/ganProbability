#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=sample
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output scripts/infogan_mnist/pixel/out_%a.out
###SBATCH --error err.txt

python sample.py --model dcgan --sampleFunc numerical --saveDir generated/final/infogan/mnist/no_ones/numerical_samples --modelroot generated/final/infogan/mnist/no_ones/ --netG generated/final/infogan/mnist/no_ones/netG_20.pth --netD generated/final/infogan/mnist/no_ones/netD_20.pth --nsamples 10000 --samplePrefix samples_${SLURM_ARRAY_TASK_ID}

