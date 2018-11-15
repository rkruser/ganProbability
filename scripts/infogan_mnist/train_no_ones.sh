#!/bin/bash
python train.py --noOnes --modelroot generated/final/infogan/mnist/no_ones --epochs 20 --model infogan --trainFunc infogan --criterion infogan --dataset mnist --noOnes
