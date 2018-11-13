#!/bin/bash
#python train.py --model DeepGAN384 --dataset cifarEmbedded384 --modelroot generated/final/deepgan --criterion gan --trainFunc gan
#python train.py --model DeepGAN10 --dataset mnistEmbedded10 --modelroot generated/final/deepgan/mnist10_improved --criterion gan --trainFunc gan --nz 10 --epochs 20
#python train.py --model DeepGAN10 --dataset cifarEmbedded10 --modelroot generated/final/deepgan/cifar10 --criterion gan --trainFunc gan --nz 10 --epochs 20
#python train.py --model DeepGAN384 --dataset mnistEmbedded384 --modelroot generated/final/deepgan/mnist384 --criterion gan --trainFunc gan --epochs 20
python train.py --model DeepGAN256 --dataset mnistAutoEmbedded --modelroot generated/final/deepgan/mnistAuto256 --criterion gan --trainFunc gan --epochs 20
