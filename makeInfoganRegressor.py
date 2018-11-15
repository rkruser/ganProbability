import torch
from models import getModels, InfoganRegressor
from train import loadModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--netG', default=None)
parser.add_argument('--netD', default=None)
parser.add_argument('--netQ', default=None)
parser.add_argument('--out', default=None)
opts = parser.parse_args()

infogan = getModels('infogan', hidden=128)
names = [opts.netG, opts.netD, opts.netQ]
loadModel(infogan, names)

infoReg = InfoganRegressor(infogan[1], infogan[2])
torch.save(infoReg.state_dict(), opts.out)




