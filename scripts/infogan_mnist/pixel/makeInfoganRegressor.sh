#!/bin/bash

python makeInfoganRegressor.py --netG generated/final/infogan/mnist/standard/netG_20.pth --netD generated/final/infogan/mnist/standard/netD_20.pth --netQ generated/final/infogan/mnist/standard/netQ_20.pth --out generated/final/infogan/mnist/standard/netR_infogan.pth
