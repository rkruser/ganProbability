#!/bin/bash
# These parameters come from the paper, but aren't much better than
#   anything else
python train.py --trainFunc wgan --modelroot generated/final/wgan_mnist/smallLRfuzzy --criterion wgan --lr 1e-4 --fuzzy --beta1 0 --beta2 0.9
