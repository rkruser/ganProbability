#!/bin/bash
python train.py --modelroot generated/final/mog_flowgan_mixed --model mogNVP --dataset mogEight --trainFunc flowgan --criterion flowgan --flowganLambda 0.5
