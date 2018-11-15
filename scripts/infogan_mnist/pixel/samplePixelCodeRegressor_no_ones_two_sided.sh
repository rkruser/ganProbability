#!/bin/bash
echo "Mnist train"
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor/netR_10.pth --samplePrefix mnistTrainTwoSidedCode --sampleFunc regressor --datamode train

echo "Mnist test"
python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor/netR_10.pth --samplePrefix mnistTestTwoSidedCode --sampleFunc regressor --datamode test

echo "Cifar train"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor/netR_10.pth --samplePrefix cifarTrainTwoSidedCode --sampleFunc regressor --datamode train

echo "Cifar test"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor/netR_10.pth --samplePrefix cifarTestTwoSidedCode --sampleFunc regressor --datamode test

echo "omniglot train"
python sample.py --model pixelRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor/netR_10.pth --samplePrefix omniglotTrainTwoSidedCode --sampleFunc regressor --datamode train

echo "omniglot test"
python sample.py --model pixelRegressor --dataset omniglot --saveDir generated/final/infogan/mnist/no_ones/data_samples  --modelroot generated/final/infogan/mnist/no_ones/ --netR generated/final/infogan/mnist/no_ones/code_prob_two_sided_regressor/netR_10.pth --samplePrefix omniglotTestTwoSidedCode --sampleFunc regressor --datamode test





