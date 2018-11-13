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
###SBATCH --error err.txt


#python sample.py --model dcgan --deepModel lenetEmbedding --sampleFunc backprop --saveDir generated/final/dcgan_mnist/samplesBackprop --samplePrefix samplesDeep --modelroot generated/final/dcgan_mnist --netG generated/final/dcgan_mnist/netG_10.pth --netD generated/final/dcgan_mnist/netD_10.pth --netEmb generated/final/lenet_mnist/netEmb_10.pth --nsamples 100
#python -m pdb sample.py --model dcgan32 --sampleFunc backprop --saveDir generated/final/dcgan_mnist/samplesBackprop --modelroot generated/final/dcgan_mnist --netG generated/final/dcgan_mnist/netG_10.pth --netD generated/final/dcgan_mnist/netD_10.pth --nsamples 10
#python -m pdb sample.py --model pixelRegressor32 --sampleFunc regressor --saveDir generated/final/dcgan_mnist/regressor --modelroot generated/final/dcgan_mnist --netG generated/final/dcgan_mnist/netG_10.pth --netD generated/final/dcgan_mnist/netD_10.pth --dataset mnist --nsamples 10 #need dataset for regressor
#python -m pdb sample.py --model lenetEmbedding32 --sampleFunc embedding --saveDir generated/final/dcgan_mnist/embeddings --modelroot generated/final/dcgan_mnist --netG generated/final/dcgan_mnist/netG_10.pth --netD generated/final/dcgan_mnist/netD_10.pth --dataset mnist --nsamples 10 #need dataset for embedding
#python sample.py --model dcgan --deepModel densenet --sampleFunc backprop --saveDir generated/final/dcgan_mnist/samplesDensenetBackprop --samplePrefix samplesDeep_${SLURM_ARRAY_TASK_ID} --modelroot generated/final/dcgan_mnist --netG generated/final/dcgan_mnist/netG_10.pth --netD generated/final/dcgan_mnist/netD_10.pth --netEmb generated/final/densenet_mnist/netEmb_10.pth --nsamples 10000
#python sample.py --model DeepGAN10 --sampleFunc backprop --saveDir generated/final/deepgan/mnist10_improved --modelroot generated/final/deepgan/mnist10_improved --netG generated/final/deepgan/mnist10_improved/netG_10.pth --netD generated/final/deepgan/mnist10_improved/netD_10.pth --nsamples 10000 --nz 10 --samplePrefix samplesDeep_${SLURM_ARRAY_TASK_ID}
#python sample.py --model DeepGAN384 --sampleFunc numerical --saveDir generated/final/deepgan/mnist384 --modelroot generated/final/deepgan/mnist384 --netG generated/final/deepgan/mnist384/netG_14.pth --netD generated/final/deepgan/mnist384/netD_14.pth --nsamples 10000 --samplePrefix samplesDeep_${SLURM_ARRAY_TASK_ID}
python sample.py --model DeepGAN256 --sampleFunc numerical --saveDir generated/final/deepgan/mnistAuto256 --modelroot generated/final/deepgan/mnistAuto256 --netG generated/final/deepgan/mnistAuto256/netG_20.pth --netD generated/final/deepgan/mnistAuto256/netD_20.pth --nsamples 10000 --samplePrefix samplesDeep_${SLURM_ARRAY_TASK_ID}




