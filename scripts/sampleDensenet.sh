#!/bin/bash
python sample.py --model densenet --dataset cifar10 --saveDir generated/final/densenet_cifar/cifar --netEmb generated/final/densenet_cifar/netEmb_6.pth --samplePrefix densenetTrainEmbedded --sampleFunc embedding --datamode train --supervised --returnEmbeddingFeats
