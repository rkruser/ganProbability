#!/bin/bash
python train.py --model lenetEmbedding --trainFunc embedding --dataset mnist --criterion softmaxbce --validation --supervised --modelroot generated/final/lenet_mnist
