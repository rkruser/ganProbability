#!/bin/bash
echo "Mnist train"
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10/netR_10.pth --samplePrefix mnistTrainEpoch10Code --sampleFunc regressor --datamode train

echo "Mnist test"
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10/netR_10.pth --samplePrefix mnistTestEpoch10Code --sampleFunc regressor --datamode test

echo "Cifar train"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10/netR_10.pth --samplePrefix cifarTrainEpoch10Code --sampleFunc regressor --datamode train

echo "Cifar test"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10/netR_10.pth --samplePrefix cifarTestEpoch10Code --sampleFunc regressor --datamode test

echo "omniglot train"
python sample.py --model pixelRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10/netR_10.pth --samplePrefix omniglotTrainEpoch10Code --sampleFunc regressor --datamode train

echo "omniglot test"
python sample.py --model pixelRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_regressor_epoch10/netR_10.pth --samplePrefix omniglotTestEpoch10Code --sampleFunc regressor --datamode test





