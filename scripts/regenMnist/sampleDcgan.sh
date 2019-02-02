#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=sample
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output scripts/regenMnist/out_%a.out
###SBATCH --error err.txt

python sample.py --model dcgan --sampleFunc numerical --saveDir $1/infogan/mnist/dcgan/numerical_samples --modelroot $1/infogan/mnist/dcgan/ --netG $1/infogan/mnist/dcgan/netG_20.pth --netD $1/infogan/mnist/dcgan/netD_20.pth --nsamples 10000 --samplePrefix samples_${SLURM_ARRAY_TASK_ID}

