#!/bin/bash
#python sample.py --model lenetEmbedding --dataset mnist --saveDir generated/final/lenet_mnist --modelroot generated/final/lenet_mnist --netEmb generated/final/lenet_mnist/netEmb_10.pth --samplePrefix mnistTestEmbedded --sampleFunc embedding --datamode test --supervised
#python sample.py --model densenet --dataset mnist --saveDir generated/final/densenet_mnist --modelroot generated/final/densenet_mnist --netEmb generated/final/densenet_mnist/netEmb_10.pth --samplePrefix mnistTestEmbedded --sampleFunc embedding --datamode test --supervised
#python sample.py --model densenet --dataset cifar10 --saveDir generated/final/densenet_cifar --modelroot generated/final/densenet_cifar --netEmb generated/final/densenet_cifar/netEmb_20.pth --samplePrefix cifarTestEmbedded --sampleFunc embedding --datamode test --supervised
#python sample.py --model densenet --returnEmbeddingFeats --dataset mnist --saveDir generated/final/densenet_mnist --modelroot generated/final/densenet_mnist --netEmb generated/final/densenet_mnist/netEmb_10.pth --samplePrefix mnistTestEmbedded384 --sampleFunc embedding --datamode test --supervised

#python sample.py --model autoencoder --dataset mnist --saveDir generated/final/mnist_autoencoder --netEnc generated/final/mnist_autoencoder/netEnc_10.pth --netDec generated/final/mnist_autoencoder/netDec_10.pth --sampleFunc embedding --datamode test --supervised --samplePrefix mnistTestEmbedded
python sample.py --model autoencoder --dataset probdata --dataroot generated/final/deepgan/mnistAuto256/samplesDeep.mat  --saveDir generated/final/deepgan/mnistAuto256 --netEnc generated/final/mnist_autoencoder/netEnc_10.pth --netDec generated/final/mnist_autoencoder/netDec_10.pth --sampleFunc sampleUp --supervised --samplePrefix ganSampleUp

