#!/bin/bash
#python train.py
python train.py --noOnes --modelroot generated/final/infogan_mnist_no_ones --epochs 20 --model infogan --trainFunc infogan --criterion infogan --dataset mnist
