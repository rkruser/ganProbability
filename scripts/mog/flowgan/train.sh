#!/bin/bash
python train.py --modelroot generated/final/mog_flowgan_seven --model mogNVP --dataset mogSeven --trainFunc flowgan --criterion flowgan --flowganLambda 0.5


# python train.py --modelroot generated/final/mog_nll --model pureNVP --dataset mogEight --trainFunc nll --epochs 1 --batchSize 100 --lr 1e-4 --checkpointEvery 1
