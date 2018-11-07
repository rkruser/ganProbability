#!/bin/bash
python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot generated/final/dcgan_mnist/samplesNumerical/samples.mat --criterion softmaxbce --validation --supervised --modelroot generated/final/dcgan_mnist/regressor
