#!/bin/bash
#python train.py --model densenet --trainFunc embedding --dataset cifar10 --criterion softmaxbce --validation --supervised --modelroot generated/final/densenet_cifar
#python train.py --model densenet --trainFunc embedding --dataset mnist --criterion softmaxbce --validation --supervised --modelroot generated/final/densenet_mnist
python train.py --model densenet --netEmb generated/final/densenet_cifar/netEmb_10.pth --epochs 20 --epochsCompleted 10 --trainFunc embedding --dataset cifar10 --criterion softmaxbce --validation --supervised --modelroot generated/final/densenet_cifar
