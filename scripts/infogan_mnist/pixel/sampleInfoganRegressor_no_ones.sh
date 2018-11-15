#!/bin/bash
echo "Mnist train"
python sample.py --model infoganRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/netR_infogan.pth --samplePrefix mnistTrainEmb --sampleFunc regressor --datamode train

echo "Mnist test"
python sample.py --model infoganRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/netR_infogan.pth --samplePrefix mnistTestEmb --sampleFunc regressor --datamode test

echo "Cifar train"
python sample.py --model infoganRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/netR_infogan.pth --samplePrefix cifarTrainEmb --sampleFunc regressor --datamode train

echo "Cifar test"
python sample.py --model infoganRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/netR_infogan.pth --samplePrefix cifarTestEmb --sampleFunc regressor --datamode test

echo "omniglot train"
python sample.py --model infoganRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/netR_infogan.pth --samplePrefix omniglotTrainEmb --sampleFunc regressor --datamode train

echo "omniglot test"
python sample.py --model infoganRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/netR_infogan.pth --samplePrefix omniglotTestEmb --sampleFunc regressor --datamode test





