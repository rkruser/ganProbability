#!/bin/bash
echo "Mnist train"
python sample.py --model infoganRegressor --dataset mnist --saveDir $1/infogan/cifar/standard/data_samples  --modelroot $1/infogan/cifar/standard/ --netR $1/infogan/cifar/standard/netR_infogan.pth --samplePrefix mnistTrainEmb --sampleFunc regressor --datamode train

echo "Mnist test"
python sample.py --model infoganRegressor --dataset mnist --saveDir $1/infogan/cifar/standard/data_samples  --modelroot $1/infogan/cifar/standard/ --netR $1/infogan/cifar/standard/netR_infogan.pth --samplePrefix mnistTestEmb --sampleFunc regressor --datamode test

echo "Cifar train"
python sample.py --model infoganRegressor --dataset cifar10 --saveDir $1/infogan/cifar/standard/data_samples  --modelroot $1/infogan/cifar/standard/ --netR $1/infogan/cifar/standard/netR_infogan.pth --samplePrefix cifarTrainEmb --sampleFunc regressor --datamode train

echo "Cifar test"
python sample.py --model infoganRegressor --dataset cifar10 --saveDir $1/infogan/cifar/standard/data_samples  --modelroot $1/infogan/cifar/standard/ --netR $1/infogan/cifar/standard/netR_infogan.pth --samplePrefix cifarTestEmb --sampleFunc regressor --datamode test

echo "omniglot train"
python sample.py --model infoganRegressor --dataset omniglot --saveDir $1/infogan/cifar/standard/data_samples  --modelroot $1/infogan/cifar/standard/ --netR $1/infogan/cifar/standard/netR_infogan.pth --samplePrefix omniglotTrainEmb --sampleFunc regressor --datamode train

echo "omniglot test"
python sample.py --model infoganRegressor --dataset omniglot --saveDir $1/infogan/cifar/standard/data_samples  --modelroot $1/infogan/cifar/standard/ --netR $1/infogan/cifar/standard/netR_infogan.pth --samplePrefix omniglotTestEmb --sampleFunc regressor --datamode test





