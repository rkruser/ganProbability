#!/bin/bash
echo "Mnist train"
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/numerical_samples/netR_25.pth --samplePrefix mnistTrainPixel --sampleFunc regressor --datamode train

echo "Mnist test"
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/numerical_samples/netR_25.pth --samplePrefix mnistTestPixel --sampleFunc regressor --datamode test

echo "Cifar train"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/numerical_samples/netR_25.pth --samplePrefix cifarTrainPixel --sampleFunc regressor --datamode train

echo "Cifar test"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/numerical_samples/netR_25.pth --samplePrefix cifarTestPixel --sampleFunc regressor --datamode test

echo "omniglot train"
python sample.py --model pixelRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/numerical_samples/netR_25.pth --samplePrefix omniglotTrainPixel --sampleFunc regressor --datamode train

echo "omniglot test"
python sample.py --model pixelRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/numerical_samples/netR_25.pth --samplePrefix omniglotTestPixel --sampleFunc regressor --datamode test





