#!/bin/bash

python train.py --modelroot generated/final/mog_infogan --model mogInfogan --dataset mogSeven --trainFunc infogan --criterion infogan --batchSize 128 --beta1 0.9


#python train.py --modelroot generated/final/mog_regular_seven --model mog_netG_improved --dataset mogSeven --trainFunc wgan --criterion wgan --batchSize 256 --beta1 0.9


# python train.py --modelroot generated/final/mog_nll --model pureNVP --dataset mogEight --trainFunc nll --epochs 1 --batchSize 100 --lr 1e-4 --checkpointEvery 1
