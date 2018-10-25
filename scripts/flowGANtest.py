# Real NVP / FlowGAN model


# Todo notes: bitmasks need a bit of reworking, since you apparently need the shape exactly right
#    Torch doesn't broadcast things
#    The scale param also has this problem, maybe?
#    Perhaps get around these problems with a built-in torch function instead of broadcasting
#    Then: test coupling layer for correctness
#          test stages for correctness
#          train realNVP

import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

#from code.total.models.nnModels import weights_init


# Todo: - finish the squeezing operations (done, except for optimizing)
#       - Redo the dimension numbers between stages so they are correct
#       - Double-check full pipeline
#       - Test untrained network on single points for consistency and invertibility / correct determinants
#       - Try training the network


# Later:
# Incorporate weight norm (easily in convStatic) and between-layer batch norm
#    (Using the module parameters running_mean, running_var, weights, bias found in
#      https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py)

# Problem: the network is so complex that it might not be comparable to DCGAN

#****** Code snipped from stackGAN
def convStatic(in_planes, out_planes, kernelSize, bias=False, weight_norm=False):
    "convolution preserving the input width and heighth"
    assert(kernelSize%2 == 1) #Must be odd
    mod = nn.Conv2d(in_planes, out_planes, kernel_size=kernelSize, stride=1,
                     padding=(kernelSize-1)/2, bias=bias)
    if weight_norm:
    	return nn.utils.weight_norm(mod)
    else:
	    return mod


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
	#   residual = x
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
			ResBlock(nhidden, kernelSize),
			ResBlock(nhidden, kernelSize),
			ResBlock(nhidden, kernelSize),
			ResBlock(nhidden, kernelSize),
	#		ResBlock(nhidden, kernelSize), #Cut down half the computation
	#		ResBlock(nhidden, kernelSize),
	#		ResBlock(nhidden, kernelSize),
	#		ResBlock(nhidden, kernelSize),
			convStatic(nhidden, nc, kernelSize, bias=biasLast) #affine output
			)

	def forward(self, x):
		return self.main(x)

class S(nn.Module):
	def __init__(self, nc, nhidden, kernelSize=3):
		super(S, self).__init__()
		self.base = T(nc, nhidden, kernelSize=kernelSize, biasLast=False)
		self.nonlinearity = nn.Tanh()
		self.scale = nn.Parameter(torch.ones(1))

	def forward(self, x):
		return self.scale*self.nonlinearity(self.base(x))


def getBitmask(size, nc, alignment):
	#gridX, gridY = torch.meshgrid([torch.arange(0,size).int(), torch.arange(0,size).int()])
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

	return bitmask.unsqueeze(0).repeat(nc,1,1).unsqueeze(0).float()



# *********

# Using ResBlocks, define S and T for 32, 16, 8, and 4 (allow for various numbers of input channels and hidden channels)
# Define the full Real NVP for 32x32 images, using S, T, Coupling, and reshaping somehow
# Form oscillating bitmasks and figure out how to do the "squeeze" described in Real NVP paper
# Check if each coupling uses different or the same S,T params
# What size should the convolutional blocks be?
# Train result using flowgan loss / log-likelihood loss and see if it works on cifar and mnist


# Take an S and T network and couple them
class Coupling(nn.Module):
	def __init__(self, S, T, bitmask=None, channelWise=None):
		super(Coupling, self).__init__()
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
		if self.bitmask is not None:
			maskedX = self.bitmask*x
			smX = self.S(maskedX)
		#	detJacob = torch.exp((self.invMask*smX).view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = (self.invMask*smX).view(smX.size(0),-1).sum(dim=1) 
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
		#	detJacob = torch.exp(smX.view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = smX.view(smX.size(0),-1).sum(dim=1)
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
		#	detJacob = 1.0/torch.exp((self.invMask*smX).view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = -(self.invMask*smX).view(smX.size(0),-1).sum(dim=1)
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
		#	detJacob = 1.0/torch.exp(smX.view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = -smX.view(smX.size(0),-1).sum(dim=1)
			x1 = y1
			x2 = (y2-self.T(x1))/smX.exp()
			if self.channelWise == 0:
				x = torch.cat((x1,x2),dim=1)
			else:
				x = torch.cat((x2,x1),dim=1)

		return x, logDetJacob #detJacob


# Note: input must have even rows/cols
def nvpSqueeze(x, horizontalIndex=None, verticalIndex=None):
	rows = x.size(2)
	cols = x.size(3)

	if horizontalIndex is None or verticalIndex is None:
		horizontalIndex = torch.cat([torch.arange(0,cols,2),torch.arange(1,cols,2)]).long().expand_as(x)
		verticalIndex = torch.cat([torch.arange(0,rows,2),torch.arange(1,rows,2)]).view(-1,1).long().expand_as(x)
#		horizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.size(0),x.size(1),x.size(2),1)
#		verticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(x.size(0),x.size(1),1,x.size(3))

		if isinstance(x,Variable):
			horizontalIndex = Variable(horizontalIndex)
			verticalIndex = Variable(verticalIndex) 

	x = torch.gather(x,3,horizontalIndex) # Need to use gather function because direct slicing isn't implemented here
	x = torch.gather(x,2,verticalIndex)

	x1 = x[:,:,:rows/2,:cols/2]
	x2 = x[:,:,:rows/2,cols/2:]
	x3 = x[:,:,rows/2:,:cols/2]
	x4 = x[:,:,rows/2:,cols/2:]
	x = torch.cat([x1,x2,x3,x4],dim=1)
	return x


def nvpUnsqueeze(y, horizontalIndex = None, verticalIndex=None):
	channels = y.size(1)

	x1 = y[:,:channels/4,:,:]
	x2 = y[:,channels/4:channels/2,:,:]
	x3 = y[:,channels/2:3*channels/4,:,:]
	x4 = y[:,3*channels/4:,:,:]	

	x12 = torch.cat([x1,x2],dim=3)
	x34 = torch.cat([x3,x4],dim=3)
	x = torch.cat([x12,x34],dim=2)

	rows = x.size(2)
	cols = x.size(3)

	if horizontalIndex is None or verticalIndex is None:
		rowperm = torch.zeros(rows)
		colperm = torch.zeros(cols)
		rowperm[torch.arange(0,cols,2).long()] = torch.arange(0,cols/2)
		rowperm[torch.arange(1,cols,2).long()] = cols/2+torch.arange(0,cols/2)
		colperm[torch.arange(0,rows,2).long()] = torch.arange(0,rows/2)
		colperm[torch.arange(1,rows,2).long()] = rows/2+torch.arange(0,rows/2)
		horizontalIndex = rowperm.long().expand_as(x)
		verticalIndex = colperm.view(-1,1).long().expand_as(x)

#		horizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.size(0),x.size(1),x.size(2),1)
#		verticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(x.size(0),x.size(1),1,x.size(3))

		if isinstance(y,Variable):
			horizontalIndex = Variable(horizontalIndex)
			verticalIndex = Variable(verticalIndex)

	x = torch.gather(x,2,verticalIndex)
	x = torch.gather(x,3,horizontalIndex)
	return x


def batchNormForward(bmod, x):
	y = bmod(x)
	channelNum = x.size(2)*x.size(3)
	logDetJacob = -0.5*(bmod.weights*(bmod.running_var+bmod.eps)*channelNum).sum()
	logDetJacob = torch.empty(x.size(0)).fill_(logDetJacob) # Same determinant for everything in batch
	return y, logDetJacob

def batchNormInvert(bmod, y):
	x = (y-bmod.bias)/bmod.weights
	x = x*torch.sqrt(bmod.running_var+bmod.eps)+bmod.running_mean #Hopefully this broadcasts right
	channelNum = x.size(2)*x.size(3)
	logDetJacob = 0.5*(bmod.weights*(bmod.running_var+bmod.eps)*channelNum).sum()
	logDetJacob = torch.empty(x.size(0)).fill_(logDetJacob) # Same determinant for everything in batch
	return x, logDetJacob


class StageType1(nn.Module):
	def __init__(self, imsize, nc, nh, ks, batchsize):
		super(StageType1, self).__init__()
		assert(imsize%2 == 0)

		self.batchsize = batchsize
		self.imsize = imsize
		self.nc = nc
		self.nh = nh
		self.ks = ks # kernel size

		# Later: can move the index initializations into here for speedup
		#  Induces a dependence on batchsize

		# forward indices
		colperm = torch.cat([torch.arange(0,imsize,2),torch.arange(1,imsize,2)]).long()
		rowperm = torch.cat([torch.arange(0,imsize,2),torch.arange(1,imsize,2)]).long()
		self.horizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchsize,nc,imsize,1)
		self.verticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batchsize,nc,1,imsize)
		self.horizontalIndex = Variable(self.horizontalIndex)
		self.verticalIndex = Variable(self.verticalIndex)

		# inverse indices
		rowperm = torch.zeros(rows)
		colperm = torch.zeros(cols)
		rowperm[torch.arange(0,imsize,2).long()] = torch.arange(0,imsize/2)
		rowperm[torch.arange(1,imsize,2).long()] = cols/2+torch.arange(0,imsize/2)
		colperm[torch.arange(0,imsize,2).long()] = torch.arange(0,imsize/2)
		colperm[torch.arange(1,imsize,2).long()] = rows/2+torch.arange(0,imsize/2)
		rowperm = rowperm.long()
		colperm = colperm.long()
		self.invHorizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchsize,nc,imsize,1)
		self.invVerticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batchsize,nc,1,imsize)
		self.invHorizontalIndex = Variable(self.invHorizontalIndex)
		self.invVerticalIndex = Variable(self.invVerticalIndex)

		self.mask = getBitmask(imsize, nc, 0)
		self.c1 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		self.b1 = nn.BatchNorm2d(nc)
		self.c2 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = 1-self.mask )
		self.b2 = nn.BatchNorm2d(nc)
		self.c3 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		self.b3 = nn.BatchNorm2d(nc)
		# Should I double the hidden layers here too?
		self.c4 = Coupling( S(2*nc, 2*nh, ks), T(2*nc, 2*nh, ks), channelWise = 1 ) # ?? Double-check redundancy
		self.b4 = nn.BatchNorm2d(2*nc)
		self.c5 = Coupling( S(2*nc, 2*nh, ks), T(2*nc, 2*nh, ks), channelWise = 0 )
		self.b5 = nn.BatchNorm2d(2*nc)
		self.c6 = Coupling( S(2*nc, 2*nh, ks), T(2*nc, 2*nh, ks), channelWise = 1 )
		self.b6 = nn.BatchNorm2d(2*nc)

	def forward(self, x):
		# Size is batchsize x nc x imsize x imsize
		out = x
		out, det1 = self.c1(out)
		out, detb1 = batchNormForward(self.b1,out)
		out, det2 = self.c2(out)
		out, detb2 = batchNormForward(self.b2,out)
		out, det3 = self.c3(out)
		out, detb3 = batchNormForward(self.b3,out)
		if self.training:
			out = nvpSqueeze(out, horizontalIndex=self.horizontalIndex, verticalIndex=self.verticalIndex)
		else:
			out = nvpSqueeze(out)
		out, det4 = self.c4(out)
		out, detb4 = batchNormForward(self.b4,out)
		out, det5 = self.c5(out)
		out, detb5 = batchNormForward(self.b5,out)
		out, det6 = self.c6(out)
		out, detb6 = batchNormForward(self.b6,out)

		logDetJacob = det1+det2+det3+det4+det5+det6+detb1+detb2+detb3+detb4+detb5+detb6

		# Out size is batchsize x (4 x nc) x (imsize/2) x (imsize/2)
		return out, logDetJacob

	def invert(self,y):
		out = y
		out, detb6 = batchNormInvert(self.b6,out)
		out, det6 = self.c6.invert(out)
		out, detb5 = batchNormInvert(self.b5,out)
		out, det5 = self.c5.invert(out)
		out, detb4 = batchNormInvert(self.b4,out)
		out, det4 = self.c4.invert(out)
		if self.training:
			out = nvpUnsqueeze(out, horizontalIndex=self.invHorizontalIndex, verticalIndex=self.invVerticalIndex)
		else:
			out = nvpUnsqueeze(out)
		out, detb3 = batchNormInvert(self.b3,out)
		out, det3 = self.c3.invert(out)
		out, detb2 = batchNormInvert(self.b2,out)
		out, det2 = self.c2.invert(out)
		out, detb1 = batchNormInvert(self.b1,out)
		out, det1 = self.c1.invert(out)

		logDetJacob = det1+det2+det3+det4+det5+det6+detb1+detb2+detb3+detb4+detb5+detb6
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
		self.b1 = nn.BatchNorm2d(nc)
		self.c2 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = 1-self.mask )
		self.b2 = nn.BatchNorm2d(nc)
		self.c3 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = self.mask )
		self.b3 = nn.BatchNorm2d(nc)
		self.c4 = Coupling( S(nc, nh, ks), T(nc, nh, ks), bitmask = 1-self.mask ) # ?? Double-check redundancy

	def forward(self, x):
		# Size is batchsize x nc x imsize x imsize
		out = x
		out, det1 = self.c1(out)
		out, detb1 = batchNormForward(self.b1,out)
		out, det2 = self.c2(out)
		out, detb2 = batchNormForward(self.b2,out)
		out, det3 = self.c3(out)
		out,detb3 = batchNormForward(self.b3,out)
		out, det4 = self.c4(out)

		logDetJacob = det1+det2+det3+det4+detb1+detb2+detb3

		# Out size is batchsize x (4 x nc) x (imsize/2) x (imsize/2)
		return out, logDetJacob

	def invert(self,y):
		out = y
		out, det4 = self.c4.invert(out)
		out, detb3 = batchNormInvert(self.b3,out)
		out, det3 = self.c3.invert(out)
		out, detb2 = batchNormInvert(self.b2,out)
		out, det2 = self.c2.invert(out)
		out, detb1 = batchNormInvert(self.b1,out)
		out, det1 = self.c1.invert(out)

		logDetJacob = det1+det2+det3+det4+detb1+detb2+detb3
		return out, logDetJacob



class RealNVP(nn.Module):
	def __init__(self, imsize, nc, nh=64, ks=3, batchsize=64):
		super(RealNVP, self).__init__()

		self.batchsize = batchsize
		self.nc = nc
		self.imsize = imsize
		self.ks = ks #for convenience below
		self.nh = nh
		self.totalSize = nc*imsize*imsize

		self.stage1 = StageType1(imsize, nc, nh, ks, batchsize)
		# factor out half
		self.stage2 = StageType1(imsize/2, 2*nc, 2*nh, ks, batchsize) 
		# Factor out half
		self.stage3 = StageType2(imsize/4, 4*nc, 4*nh, ks) # nc will overtake nh unless nh starts high enough

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
		z1z2 = z1z2.view(z.size(0), 2*self.nc, self.imsize/2, self.imsize/2)
		z3 = z[:,self.totalSize/2:3*self.totalSize/4]
		z3 = z3.view(z.size(0), 4*self.nc, self.imsize/4, self.imsize/4)
		z4 = z[:,3*self.totalSize/4:]
		z4 = z4.view(z.size(0), 4*self.nc, self.imsize/4, self.imsize/4)

		z4in, det3 = self.stage3.invert(z4)
		z3 = torch.cat((z3,z4in),dim=1)
		z3in, det2 = self.stage2.invert(z3)
		z1z2 = torch.cat((z1z2, z3in), dim=1)
		x, det1 = self.stage1.invert(z1z2)

		logDetJacob = det1+det2+det3
		return x, logDetJacob




################################################################3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.normal(m.weight, std=1e-2)
        # init.orthogonal(m.weight)
        #init.xavier_uniform(m.weight, gain=1.4)
        if m.bias is not None:
            init.constant(m.bias, 0.0)

def test(variable=False):
	a = torch.arange(0,144).resize_((2,2,6,6))#.unsqueeze(0)
	if variable:
		a = Variable(a)
	b = nvpSqueeze(a)
	c = nvpUnsqueeze(b)

	print a
	print b
	print c

def testBitmask():
	a = getBitmask(6,3,0)
	b = getBitmask(6,3,1)

	print a, b

def testCoupling():
	mask = Variable(getBitmask(6,3,0))
	c = Coupling( S(3, 5), T(3,5), bitmask=mask )
	c.apply(weights_init)

	a = torch.zeros(2,3,6,6)
	a.normal_()
	a = Variable(a)
	y = c(a)
	yinv = c.invert(y)

	print a, yinv, y


if __name__=='__main__':
	test(variable=True)
#	testBitmask()
#	testCoupling()