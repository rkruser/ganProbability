#!/bin/bash

python makeInfoganRegressor.py --netG $1/infogan/cifar/standard/netG_20.pth --netD $1/infogan/cifar/standard/netD_20.pth --netQ $1/infogan/cifar/standard/netQ_20.pth --out $1/infogan/cifar/standard/netR_infogan.pth
