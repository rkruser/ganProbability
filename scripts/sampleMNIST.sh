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



#python sample.py --model dcgan --saveDir generated/final/dcgan_mnist_no_ones --netG generated/final/dcgan_mnist_no_ones/netG_20.pth --netD generated/final/dcgan_mnist_no_ones/netD_20.pth --samplePrefix samples_${SLURM_ARRAY_TASK_ID} --sampleFunc numerical --nsamples 10000

#python sample.py --model pixelRegressor --saveDir generated/final/dcgan_mnist_no_ones/mnistFullSample --netR generated/final/dcgan_mnist_no_ones/netR_25.pth --samplePrefix mnistFullSample  --sampleFunc regressor --dataset mnist --datamode train

#python sample.py --model pixelRegressor --saveDir generated/final/dcgan_mnist_no_ones/cifarSample --netR generated/final/dcgan_mnist_no_ones/netR_25.pth --samplePrefix cifarUnderMnist  --sampleFunc regressor --dataset cifar10 --datamode train

#python sample.py --model pixelRegressor --saveDir generated/final/dcgan_mnist_no_ones/omniglotSample --netR generated/final/dcgan_mnist_no_ones/netR_25.pth --samplePrefix omniglotUnderMnist  --sampleFunc regressor --dataset omniglot --datamode train

#python sample.py --model pixelRegressor --saveDir generated/final/dcgan_mnist_no_ones/twoSided/omniglot --netR generated/final/dcgan_mnist_no_ones/twoSided/netR_25.pth --samplePrefix omniglot  --sampleFunc regressor --dataset omniglot --datamode train

#python sample.py --model pixelRegressor --saveDir generated/final/dcgan_mnist_no_ones/twoSided/mnist --netR generated/final/dcgan_mnist_no_ones/twoSided/netR_25.pth --samplePrefix mnist  --sampleFunc regressor --dataset mnist --datamode train

python sample.py --model pixelRegressor --saveDir generated/final/dcgan_mnist_no_ones/fuzzy/ --netR generated/final/dcgan_mnist_no_ones/netR_25.pth --samplePrefix mnist  --sampleFunc regressor --dataset mnist --datamode train --fuzzy







