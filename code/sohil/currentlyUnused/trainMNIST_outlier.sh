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
time python pretraining.py --dataset outlier2 --outf /fs/vulcan-scratch/krusinga/externalProjects/sohilGAN/ProbDistGAN/ryenExperiments/outputs/mnist_outlier_z_10_epoch_25 --nc 1 --niter 25 --dataroot /vulcan/scratch/krusinga/mnist --nz 10 --imageSize 28 --cuda --proportions '[0.02,0.98/9, 0.98/9, 0.98/9, 0.98/9, 0.98/9, 0.98/9, 0.98/9, 0.98/9, 0.98/9, 0.98/9]'

