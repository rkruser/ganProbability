#!/bin/bash
python train.py --modelroot generated/final/mog_flowgan --model mogNVP --dataset mogEight --trainFunc flowgan --criterion flowgan --flowganLambda 1.0
