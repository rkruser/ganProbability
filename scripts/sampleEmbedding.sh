#!/bin/bash
python sample.py --model lenetEmbedding --dataset mnist --saveDir generated/final/lenet_mnist --modelroot generated/final/lenet_mnist --netEmb generated/final/lenet_mnist/netEmb_10.pth --samplePrefix mnistTestEmbedded --sampleFunc embedding --datamode test --supervised

