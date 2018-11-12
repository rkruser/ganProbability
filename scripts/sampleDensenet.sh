#!/bin/bash
python sample.py --model densenet --useLargerEmbedding --dataset cifar10 --saveDir generated/final/densenet_cifar/cifar --netEmb generated/final/densenet_cifar/netEmb_14.pth --samplePrefix densenetTestEmbedded --sampleFunc embedding --datamode test --supervised --returnEmbeddingFeats
