#!/bin/bash
python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot generated/final/infogan/mnist/no_ones/numerical_samples_epoch10/samples.mat --criterion l2 --validation --supervised --modelroot generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10 --lr 1e-4 --checkpointEvery 5 --epochs 25 --useCodeProbs
