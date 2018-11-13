#!/bin/bash
#python sample.py --model pixelRegressor --dataset mnist --saveDir generated/final/dcgan_mnist/regressor --modelroot generated/final/dcgan_mnist/regressor --netR generated/final/dcgan_mnist/regressor/netR_10.pth --samplePrefix mnistThroughRegressorTest --sampleFunc regressor --datamode test
#python sample.py --model DeepRegressor10 --dataset embeddedMnist --dataroot generated/final/densenet_mnist/mnistEmbedded.mat  --saveDir generated/final/dcgan_mnist/samplesDensenetBackprop/rsamples --modelroot generated/final/dcgan_mnist/samplesDensenetBackprop --netR generated/final/dcgan_mnist/samplesDensenetBackprop/netR_50.pth --samplePrefix mnistViaDeepRegressorTest --sampleFunc regressor --datamode train
#python sample.py --model DeepRegressor10 --dataset mnistEmbedded10 --saveDir generated/final/deepgan/mnist10_improved/rsamples --modelroot generated/final/deepgan/mnist10_improved --netR generated/final/deepgan/mnist10_improved/netR_50.pth --samplePrefix mnistViaDeepRegressor --sampleFunc regressor --datamode train
#python sample.py --model DeepRegressor10 --dataset cifarEmbedded10 --saveDir generated/final/deepgan/cifar10/rsample --modelroot generated/final/deepgan/cifar10 --netR generated/final/deepgan/cifar10/netR_50.pth --samplePrefix cifarViaDeepRegressor --sampleFunc regressor --datamode train
#python sample.py --model DeepRegressor384 --dataset mnistEmbedded384 --saveDir generated/final/deepgan/mnist384/rsample --modelroot generated/final/deepgan/mnist384 --netR generated/final/deepgan/mnist384/netR_50.pth --samplePrefix mnistViaDeepRegressorTest --sampleFunc regressor --datamode test
python sample.py --model DeepRegressor256 --dataset mnistAutoEmbedded --saveDir generated/final/deepgan/mnistAuto256/rsample --modelroot generated/final/deepgan/mnistAuto256 --netR generated/final/deepgan/mnistAuto256/netR_50.pth --samplePrefix mnistViaAutoencoderTest --sampleFunc regressor --datamode test




