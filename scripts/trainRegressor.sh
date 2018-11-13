#!/bin/bash
#python train.py --model pixelRegressor --trainFunc regressor --dataset probdata --dataroot generated/final/dcgan_mnist/samplesNumerical/samples.mat --criterion softmaxbce --validation --supervised --modelroot generated/final/dcgan_mnist/regressor
#python train.py --model DeepRegressor10 --trainFunc regressor --dataset probdata --dataroot generated/final/dcgan_mnist/samplesDensenetBackprop/samples.mat --criterion l2 --validation --supervised --modelroot generated/final/dcgan_mnist/samplesDensenetBackprop --epochs 50 --lr 1e-4 --checkpointEvery 5
#python train.py --model DeepRegressor10 --trainFunc regressor --dataset probdata --dataroot generated/final/deepgan/mnist10_improved/samplesDeep.mat --criterion l2 --validation --supervised --modelroot generated/final/deepgan/mnist10_improved --epochs 50 --lr 1e-4 --checkpointEvery 5
#python train.py --model DeepRegressor10 --trainFunc regressor --dataset probdata --dataroot generated/final/deepgan/cifar10/samplesDeep.mat --criterion l2 --validation --supervised --modelroot generated/final/deepgan/cifar10 --epochs 50 --lr 1e-4 --checkpointEvery 5
#python train.py --model DeepRegressor384 --trainFunc regressor --dataset probdata --dataroot generated/final/deepgan/mnist384/samplesDeep.mat --criterion l2 --validation --supervised --modelroot generated/final/deepgan/mnist384 --epochs 50 --lr 1e-4 --checkpointEvery 5
python train.py --model DeepRegressor256 --trainFunc regressor --dataset probdata --dataroot generated/final/deepgan/mnistAuto256/samplesDeep.mat --criterion l2 --validation --supervised --modelroot generated/final/deepgan/mnistAuto256 --epochs 50 --lr 1e-4 --checkpointEvery 5
