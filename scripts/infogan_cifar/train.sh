#!/bin/bash
python train.py --noOnes --modelroot generated/final/infogan/cifar --epochs 20 --model infogan --trainFunc infogan --criterion infogan --dataset cifar10
