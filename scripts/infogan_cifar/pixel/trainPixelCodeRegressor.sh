#!/bin/bash
python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot generated/final/infogan/cifar/standard/numerical_samples/samples.mat --criterion l2 --validation --supervised --modelroot generated/final/infogan/cifar/standard/code_prob_regressor --lr 1e-4 --checkpointEvery 5 --epochs 25 --useCodeProbs
