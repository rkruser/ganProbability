#!/bin/bash
python train.py --modelroot generated/final/mnist_autoencoder --model autoencoder --trainFunc autoencoder --criterion l2 --dataset mnist
