#!/bin/bash
python train.py --noOnes --modelroot generated/final/infogan/mnist/standard --epochs 20 --model infogan --trainFunc infogan --criterion infogan --dataset mnist 
