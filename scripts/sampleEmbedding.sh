#!/bin/bash
#python sample.py --model lenetEmbedding --dataset mnist --saveDir generated/final/lenet_mnist --modelroot generated/final/lenet_mnist --netEmb generated/final/lenet_mnist/netEmb_10.pth --samplePrefix mnistTestEmbedded --sampleFunc embedding --datamode test --supervised
#python sample.py --model densenet --dataset mnist --saveDir generated/final/densenet_mnist --modelroot generated/final/densenet_mnist --netEmb generated/final/densenet_mnist/netEmb_10.pth --samplePrefix mnistTestEmbedded --sampleFunc embedding --datamode test --supervised
python sample.py --model densenet --dataset cifar10 --saveDir generated/final/densenet_cifar --modelroot generated/final/densenet_cifar --netEmb generated/final/densenet_cifar/netEmb_20.pth --samplePrefix cifarTestEmbedded --sampleFunc embedding --datamode test --supervised


