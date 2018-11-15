#!/bin/bash

#SBATCH --array=0-9
#SBATCH --job-name=sample
###SBATCH --qos=default
#SBATCH --mem=16gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output out_%a.out


# This one ****
#python sample.py --model infoganRegressor --saveDir generated/final/infogan_mnist_no_ones/mnist --netR generated/final/infogan_mnist_no_ones/netR_infogan.pth --samplePrefix mnist  --sampleFunc regressor --dataset mnist --datamode train

#python sample.py --model infoganRegressor --saveDir generated/final/infogan_mnist_no_ones/omniglot --netR generated/final/infogan_mnist_no_ones/netR_infogan.pth --samplePrefix omniglot  --sampleFunc regressor --dataset omniglot --datamode train

#python sample.py --model dcgan --saveDir generated/final/infogan_mnist_no_ones/samples --netG generated/final/infogan_mnist_no_ones/netG_20.pth --netD generated/final/infogan_mnist_no_ones/netD_20.pth --samplePrefix sample_${SLURM_ARRAY_TASK_ID} --sampleFunc numerical --nsamples 10000

python sample.py --model pixelRegressor --saveDir generated/final/infogan_mnist_no_ones/mnistPixel --netR generated/final/infogan_mnist_no_ones/netR_25.pth --sampleFunc regressor --dataset mnist --datamode train --samplePrefix mnist








