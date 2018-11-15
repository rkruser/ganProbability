#!/bin/bash

python makeInfoganRegressor.py --netG generated/final/infogan/cifar/standard/netG_20.pth --netD generated/final/infogan/cifar/standard/netD_20.pth --netQ generated/final/infogan/cifar/standard/netQ_20.pth --out generated/final/infogan/cifar/standard/netR_infogan.pth
