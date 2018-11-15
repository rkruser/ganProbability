#!/bin/bash

python makeInfoganRegressor.py --netG generated/final/infogan/mnist/no_ones/netG_20.pth --netD generated/final/infogan/mnist/no_ones/netD_20.pth --netQ generated/final/infogan/mnist/no_ones/netQ_20.pth --out generated/final/infogan/mnist/no_ones/netR_infogan.pth
