#!/bin/bash
python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot generated/final/infogan/mnist/no_ones/numerical_samples/samples.mat --criterion smoothL1TwoSided --validation --supervised --modelroot generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor --lr 1e-4 --checkpointEvery 5 --epochs 10 --useCodeProbs
