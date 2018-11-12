#!/bin/bash
python train.py --model DeepGAN384 --dataset cifarEmbedded384 --modelroot generated/final/deepgan --criterion gan --trainFunc gan
