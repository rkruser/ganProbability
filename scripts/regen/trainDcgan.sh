#!/bin/bash
python train.py --noOnes --modelroot $1/infogan/cifar/dcgan --epochs 20 --model dcgan --trainFunc gan --criterion gan --dataset cifar10
