# Real NVP / FlowGAN model
from mlworkflow import Data
from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init

import argparse

import torch
import torch.nn as nn


#****** Code snipped from stackGAN
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out
# *********

# Using Resblocks, define S and T for 32, 16, 8, and 4 (allow for various numbers of input channels and hidden channels)
# Define the full Real NVP for 32x32 images, using S, T, Coupling, and reshaping somehow
# Form oscillating bitmasks and figure out how to do the "squeeze" described in Real NVP paper
# Check if each coupling uses different or the same S,T params
# What size should the convolutional blocks be?
# Train result using flowgan loss / log-likelihood loss and see if it works on cifar and mnist


# Take an S and T network and couple them
class Coupling(nn.Module):
	def __init__(self, bitmask, S, T):
		self.bitmask = bitmask
		self.invMask = 1-bitmask
		self.S = S
		self.T = T

	def forward(self, x):
		maskedX = self.bitmask*x
		smX = self.S(maskedX)
		detJacob = torch.exp((self.invMask*smX).view(smX.size()[0],-1).sum(dim=1))
		y = maskedX + self.invMask*(x*smX.exp()+self.T(maskedX))
		return y, detJacob

	def invert(self, y):
		maskedX = self.bitmask*y # equivalent to bitmask*x
		smX = self.S(maskedX)
		detJacob = 1.0/torch.exp((self.invMask*smX).view(smX.size()[0],-1).sum(dim=1))
		x = self.maskedX + self.invMask*((y-self.T(maskedX))/smX.exp())
		return x, detJacob




class realNVP32(nn.Module):
	def __init__(self):
		super(realNVP, self).__init__()

		# Need s and t networks, coupling layers

		# Need ability to bitmask and to alternate patterns



class FlowGANModel(ModelTemplate):
	def __init__(self, config, args):
		super(FlowGANModel, self).__init__(config, args)





def test():
	parser = argparse.ArgumentParser()
	parser.parse_args()



if __name__=='__main__':
	test()
