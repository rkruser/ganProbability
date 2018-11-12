#!/bin/bash
python train.py --modelroot generated/final/mog_nll --model pureNVP --dataset mogEight --trainFunc nll --epochs 1 --batchSize 100 --lr 1e-4 --checkpointEvery 1
