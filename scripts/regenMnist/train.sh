#!/bin/bash
python train.py --noOnes --modelroot $1/infogan/mnist/dcgan --epochs 20 --model infogan --trainFunc infogan --criterion infogan --dataset mnist
