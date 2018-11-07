#!/bin/bash
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/dcgan_mnist/regressor --modelroot generated/final/dcgan_mnist/regressor --netR generated/final/dcgan_mnist/regressor/netR_10.pth --samplePrefix mnistThroughRegressorTest --sampleFunc regressor --datamode test

