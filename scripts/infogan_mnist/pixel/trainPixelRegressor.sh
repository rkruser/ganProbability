#!/bin/bash
python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot generated/final/infogan/mnist/standard/numerical_samples/samples.mat --criterion l2 --validation --supervised --modelroot generated/final/infogan/mnist/standard/numerical_samples --lr 1e-4 --checkpointEvery 5 --epochs 25 
