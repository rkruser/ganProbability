#!/bin/bash
#python train.py --modelroot generated/final/mnist_autoencoder --model autoencoder --trainFunc autoencoder --criterion l2 --dataset mnist
python train.py --modelroot generated/final/cifar_autoencoder --model autoencoder2 --trainFunc autoencoder --criterion l2 --dataset cifar10 --lr 0.00001
