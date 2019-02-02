#!/bin/bash
python train.py --noOnes --modelroot $1/infogan/cifar/standard --epochs 20 --model infogan --trainFunc infogan --criterion infogan --dataset cifar10
