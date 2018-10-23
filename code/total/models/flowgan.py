# Real NVP / FlowGAN model
from mlworkflow import Data
from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init

import argparse

import torch
import torch.nn as nn

# Todo: - finish the squeezing operations
#       - Redo the dimension numbers between stages so they are correct
#       - Double-check full pipeline
#       - Test untrained network on single points for consistency and invertibility / correct determinants
#       - Try training the network


#****** Code snipped from stackGAN
def convStatic(in_planes, out_planes, kernelSize, bias=False):
    "convolution preserving the input width and heighth"
    assert(kernelSize%2 == 1) #Must be odd
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelSize, stride=1,
                     padding=(kernelSize-1)/2, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, channel_num, kernelSize):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            convStatic(channel_num, channel_num, kernelSize),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            convStatic(channel_num, channel_num, kernelSize),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(True)

    def forward(self, x):
#        residual = x
        out = self.block(x)
        out += x
        out = self.relu(out)
        return out

class T(nn.Module):
	def __init__(self, nc, nhidden, kernelSize=3, biasLast=True):
		super(T, self).__init__()
		self.main = nn.Sequential(
			convStatic(nc, nhidden, kernelSize),
			nn.BatchNorm2d(nhidden),
			nn.ReLU(True),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			Resblock(nhidden, kernelSize),
			convStatic(nhidden, nc, kernelSize, bias=biasLast)
			)

	def forward(self, x):
		return self.main(x)

class S(nn.Module):
	def __init__(self, nc, nhidden, kernelSize=3):
		super(S, self).__init__()
		self.base = T(nc, nhidden, kernelSize=kernelSize, biasLast=False)
		self.nonlinearity = nn.Tanh()

	def forward(self, x):
		return self.nonlinearity(self.base(x))


def getBitmask(size, nc, alignment):
#	gridX, gridY = torch.meshgrid([torch.arange(0,size).int(), torch.arange(0,size).int()])
	xrange = torch.arange(0,size).int()
	yrange = torch.arange(0,size).int()

	xgrid = xrange.unsqueeze(1).repeat(1,size)
	ygrid = yrange.unsqueeze(0).repeat(size,1)
	total = xgrid+ygrid

	bitmask = None
	if alignment == 0:
		bitmask = (total%2 == 0)
	else:
		bitmask = (total%2 == 1)

	return bitmask.unsqueeze(2).repeat(1,1,nc).float()



# *********

# Using Resblocks, define S and T for 32, 16, 8, and 4 (allow for various numbers of input channels and hidden channels)
# Define the full Real NVP for 32x32 images, using S, T, Coupling, and reshaping somehow
# Form oscillating bitmasks and figure out how to do the "squeeze" described in Real NVP paper
# Check if each coupling uses different or the same S,T params
# What size should the convolutional blocks be?
# Train result using flowgan loss / log-likelihood loss and see if it works on cifar and mnist


# Take an S and T network and couple them
class Coupling(nn.Module):
	def __init__(self, S, T, bitmask=None, channelWise=None):
		assert((bitmask is not None) or (channelWise is not None))
		self.bitmask = bitmask
		if bitmask is not None:
			self.invMask = 1-bitmask
		if channelWise is not None:
			assert(channelWise == 0 or channelWise == 1)
		self.channelWise = channelWise

		self.S = S
		self.T = T

	def forward(self, x):
		if bitmask is not None:
			maskedX = self.bitmask*x
			smX = self.S(maskedX)
#			detJacob = torch.exp((self.invMask*smX).view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = (self.invMask*smX).view(smX.size()[0],-1).sum(dim=1) 
			y = maskedX + self.invMask*(x*smX.exp()+self.T(maskedX))
		else:
			ch1, ch2 = torch.chunk(x, 2, dim=1)
			if self.channelWise == 0:
				x1 = ch1
				x2 = ch2
			else:
				x1 = ch2
				x2 = ch1
			smX = self.S(x1)
#			detJacob = torch.exp(smX.view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = smX.view(smX.size()[0],-1).sum(dim=1)
			y1 = x1
			y2 = x2*smX.exp()+self.T(x1)
			if self.channelWise == 0:
				y = torch.cat((y1,y2),dim=1)
			else:
				y = torch.cat((y2,y1),dim=1)

		return y, logDetJacob #detJacob

	def invert(self, y):
		if self.bitmask is not None:
			maskedX = self.bitmask*y # equivalent to bitmask*x
			smX = self.S(maskedX)
#			detJacob = 1.0/torch.exp((self.invMask*smX).view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = -(self.invMask*smX).view(smX.size()[0],-1).sum(dim=1)
			x = self.maskedX + self.invMask*((y-self.T(maskedX))/smX.exp())
		else:
			ch1, ch2 = torch.chunk(y, 2, dim=1)
			if self.channelWise == 0:
				y1 = ch1
				y2 = ch2
			else:
				y1 = ch2
				y2 = ch1
			smX = self.S(y1)
#			detJacob = 1.0/torch.exp(smX.view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = -smX.view(smX.size()[0],-1).sum(dim=1)
			x1 = y1
			x2 = (y2-self.T(x1))/smX.exp()
			if self.channelWise == 0:
				x = torch.cat((x1,x2),dim=1)
			else:
				x = torch.cat((x2,x1),dim=1)

		return x, logDetJacob #detJacob


def nvpSqueeze(x, mask):
	imsize = x.size(2)
	white = x[mask.int()].view(x.size(0),x.size(1),imsize,imsize/2)
	black = x[1-mask.int()].view(x.size(0),x.size(1),imsize, imsize/2)
	# Gather even rows of white/black into one side, odd rows into another
	# ....

def nvpUnsqueeze(y, mask):
	pass

class StageType1(nn.Module):
	def __init__(self, imsize, nc, nh, ks):
		super(StageType1, self).__init__()
		self.imsize = imsize
		self.nc = nc
		self.nh = nh
		self.ks = ks #for convenience below

		self.mask = getBitmask(imsize, nc, 0)
		self.c1 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		self.c2 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = 1-self.mask )
		self.c3 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		# Should I double the hidden layers here too?
		self.c4 = Coupling( S(2*nc, nh, ks), T(2*nc, nh, ks), channelWise = 1 ) # ?? Double-check redundancy
		self.c5 = Coupling( S(2*nc, nh, ks), T(2*nc, nh, ks), channelWise = 0 )
		self.c6 = Coupling( S(2*nc, nh, ks), T(2*nc, nh, ks), channelWise = 1 )

	def forward(self, x):
		# Size is batchsize x nc x imsize x imsize
		out = x
		out, det1 = self.c1(out)
		out, det2 = self.c2(out)
		out, det3 = self.c3(out)
		out = nvpSqueeze(out,self.mask)
		out, det4 = self.c4(out)
		out, det5 = self.c5(out)
		out, det6 = self.c6(out)

		logDetJacob = det1+det2+det3+det4+det5+det6

		# Out size is batchsize x (4 x nc) x (imsize/2) x (imsize/2)
		return out, logDetJacob

	def invert(self,y):
		out = y
		out, det6 = self.c6.invert(out)
		out, det5 = self.c5.invert(out)
		out, det4 = self.c4.invert(out)
		out = nvpUnsqueeze(out,self.mask)
		out, det3 = self.c3.invert(out)
		out, det2 = self.c2.invert(out)
		out, det1 = self.c1.invert(out)

		logDetJacob = det1+det2+det3+det4+det5+det6
		return out, logDetJacob


class StageType2(nn.Module):
	def __init__(self, imsize, nc, nh, ks):
		super(StageType2, self).__init__()
		self.imsize = imsize
		self.nc = nc
		self.nh = nh
		self.ks = ks #for convenience below

		self.mask = getBitmask(imsize, nc, 0)
		self.c1 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		self.c2 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = 1-self.mask )
		self.c3 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		self.c4 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = 1-self.mask ) # ?? Double-check redundancy

	def forward(self, x):
		# Size is batchsize x nc x imsize x imsize
		out = x
		out, det1 = self.c1(out)
		out, det2 = self.c2(out)
		out, det3 = self.c3(out)
		out, det4 = self.c4(out)

		logDetJacob = det1+det2+det3+det4

		# Out size is batchsize x (4 x nc) x (imsize/2) x (imsize/2)
		return out, logDetJacob

	def invert(self,y):
		out = y
		out, det4 = self.c4.invert(out)
		out, det3 = self.c3.invert(out)
		out, det2 = self.c2.invert(out)
		out, det1 = self.c1.invert(out)

		logDetJacob = det1+det2+det3+det4
		return out, logDetJacob



class RealNVP(nn.Module):
	def __init__(self, imsize, nc, nh=64, ks=3):
		super(RealNVP, self).__init__()

		self.nc = nc
		self.imsize = imsize
		self.ks = ks #for convenience below
		self.nh = nh
		self.totalSize = nc*imsize*imsize


		self.stage1 = StageType1(imsize, nc, nh, ks)
		# factor out half
		# ** The numbers are wrong, pls fix
		self.stage2 = StageType1(imsize/2, 4*nc, 2*nh, ks)
		# Factor out half
		self.stage3 = StageType2(imsize/4, 16*nc, 4*nh, ks)

		# Need s and t networks, coupling layers

		# Need ability to bitmask and to alternate patterns

	def forward(self, x):
		out = x
		out, det1 = self.stage1(out)
		z1z2, out = torch.chunk(out,2,dim=1)
		out, det2 = self.stage2(out)
		z3, out = torch.chunk(out,2,dim=1)
		z4, det3 = self.stage3(out)

		logDetJacob = det1+det2+det3
		b = x.size(0)
		z = torch.cat((z1z2.view(b,-1), z3.view(b,-1), z4.view(b,-1)), dim=1)

		return z, logDetJacob

	def invert(self, z):
		z1z2 = z[:,:self.totalSize/2]
		# The numbers are wrong; pls fix
		z1z2 = z1z2.view(-1, 2*self.nc, self.imsize/2, self.imsize/2)
		z3 = z[:,(self.totalSize/2+1):(3*self.totalSize/4)]
		z3 = z3.view(-1, 8*self.nc, self.imsize/4, self.imsize/4)
		z4 = z[:,(3*self.totalSize/4+1):]
		z4 = z4.view(-1, 8*self.nc, self.imsize/4, self.imsize/4)

		z4in, det3 = self.stage3.invert(z4)
		z3 = torch.cat((z3,z4in),dim=1)
		z3in, det2 = self.stage2.invert(z3)
		z1z2 = torch.cat((z1z2, z3in), dim=1)
		x, det1 = self.stage1.invert(z1z2)

		logDetJacob = det1+det2+det3
		return x, logDetJacob







class FlowGANModel(ModelTemplate):
	def __init__(self, config, args):
		super(FlowGANModel, self).__init__(config, args)





def test():
	parser = argparse.ArgumentParser()
	parser.parse_args()



if __name__=='__main__':
	test()
