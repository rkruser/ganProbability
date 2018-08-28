#!/bin/bash
# Train MNIST model

###SBATCH --array=0-9
#SBATCH --job-name=mnist
###SBATCH --qos=default
#SBATCH --account=scavenger
#SBATCH --partition scavenger
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
###SBATCH --output ../output/samples/samples3/out-%A-%a.txt
###SBATCH --error ../output/samples/samples3/err-%A-%a.txt

module load cuda
time python pretraining.py --dataset mnist --outf /fs/vulcan-scratch/krusinga/externalProjects/sohilGAN/ProbDistGAN/ryenExperiments/outputs/ --nc 1 --niter 50 --dataroot /vulcan/scratch/krusinga/mnist --nz 20 --imageSize 28 --cuda
