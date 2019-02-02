#!/bin/bash

python zip.py --zipsamples --nfiles 10 --samplePrefix $1/infogan/mnist/dcgan/numerical_samples/samples --out $1/infogan/mnist/dcgan/numerical_samples/samples.mat
