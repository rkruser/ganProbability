#!/bin/bash
python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot $1/infogan/cifar/dcgan/numerical_samples/samples.mat --criterion l2 --validation --supervised --modelroot $1/infogan/cifar/dcgan/code_prob_regressor --lr 1e-4 --checkpointEvery 5 --epochs 25 --useCodeProbs
