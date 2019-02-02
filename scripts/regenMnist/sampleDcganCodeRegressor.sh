#!/bin/bash
echo "Mnist train"
python sample.py --model pixelRegressor --dataset mnist --saveDir $1/infogan/mnist/dcgan/data_samples  --modelroot $1/infogan/mnist/dcgan/ --netR $1/infogan/mnist/dcgan/code_prob_regressor/netR_25.pth --samplePrefix mnistTrainCode --sampleFunc regressor --datamode train

echo "Mnist test"
python sample.py --model pixelRegressor --dataset mnist --saveDir $1/infogan/mnist/dcgan/data_samples  --modelroot $1/infogan/mnist/dcgan/ --netR $1/infogan/mnist/dcgan/code_prob_regressor/netR_25.pth --samplePrefix mnistTestCode --sampleFunc regressor --datamode test

echo "Cifar train"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir $1/infogan/mnist/dcgan/data_samples  --modelroot $1/infogan/mnist/dcgan/ --netR $1/infogan/mnist/dcgan/code_prob_regressor/netR_25.pth --samplePrefix cifarTrainCode --sampleFunc regressor --datamode train

echo "Cifar test"
python sample.py --model pixelRegressor --dataset cifar10 --saveDir $1/infogan/mnist/dcgan/data_samples  --modelroot $1/infogan/mnist/dcgan/ --netR $1/infogan/mnist/dcgan/code_prob_regressor/netR_25.pth --samplePrefix cifarTestCode --sampleFunc regressor --datamode test

echo "omniglot train"
python sample.py --model pixelRegressor --dataset omniglot --saveDir $1/infogan/mnist/dcgan/data_samples  --modelroot $1/infogan/mnist/dcgan/ --netR $1/infogan/mnist/dcgan/code_prob_regressor/netR_25.pth --samplePrefix omniglotTrainCode --sampleFunc regressor --datamode train

echo "omniglot test"
python sample.py --model pixelRegressor --dataset omniglot --saveDir $1/infogan/mnist/dcgan/data_samples  --modelroot $1/infogan/mnist/dcgan/ --netR $1/infogan/mnist/dcgan/code_prob_regressor/netR_25.pth --samplePrefix omniglotTestCode --sampleFunc regressor --datamode test





