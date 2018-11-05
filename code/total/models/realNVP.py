# Real NVP / FlowGAN model

# Next todo (10/26):
#  Add gpu clauses to RealNVP class
#  Incorporate into mlworkflow setup
#  train model with flowGAN loss

# More todo:
#   Make batchnorm functions nice and clean, with caching
#   Overload eval function to go to running mean during evaluation (maybe)

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
def convStatic(in_planes, out_planes, kernelSize, bias=False, weight_norm=True):
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
		return self.scale.expand_as(x)*self.nonlinearity(self.base(x)) #Note: expand_as used here, this should be right


def getBitmask(size, nc, batchsize, alignment):
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

	return bitmask.unsqueeze(0).unsqueeze(0).repeat(batchsize,nc,1,1).float()



# *********

# Using ResBlocks, define S and T for 32, 16, 8, and 4 (allow for various numbers of input channels and hidden channels)
# Define the full Real NVP for 32x32 images, using S, T, Coupling, and reshaping somehow
# Form oscillating bitmasks and figure out how to do the "squeeze" described in Real NVP paper
# Check if each coupling uses different or the same S,T params
# What size should the convolutional blocks be?
# Train result using flowgan loss / log-likelih ood loss and see if it works on cifar and mnist


# Take an S and T network and couple them
class Coupling(nn.Module):
	def __init__(self, S, T, channelWise=None, align=0):
		super(Coupling, self).__init__()
		if channelWise is not None:
			assert(channelWise == 0 or channelWise == 1)
		self.channelWise = channelWise
		self.align = align

		self.S = S
		self.T = T

	def forward(self, x, bitmask=None, invBitmask=None):
		if self.channelWise is None:
			if bitmask is None:
				# Suggested optimization: use expand_as here and call getBitmask in __init__ for a plane template
				bitmask = Variable(getBitmask(x.size(2),x.size(1),x.size(0),self.align))
				invBitmask = 1-bitmask
				if x.is_cuda:
					bitmask = bitmask.cuda()
					invBitmask = invBitmask.cuda()

			maskedX = bitmask*x
			smX = self.S(maskedX)
		#	detJacob = torch.exp((self.invBitmask*smX).view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = (invBitmask*smX).view(smX.size(0),-1).sum(dim=1) 
			y = maskedX + invBitmask*(x*smX.exp()+self.T(maskedX))
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

	def invert(self, y, bitmask=None, invBitmask=None):
		if self.channelWise is None:
			if bitmask is None:
				# Suggested optimization: use expand_as here and call getBitmask in __init__ for a plane template
				bitmask = Variable(getBitmask(y.size(2),y.size(1),y.size(0),self.align))
				invBitmask = 1-bitmask
				if y.is_cuda:
					bitmask = bitmask.cuda()
					invBitmask = invBitmask.cuda()
			maskedX = bitmask*y # equivalent to bitmask*x
			smX = self.S(maskedX)
		#	detJacob = 1.0/torch.exp((self.invBitmask*smX).view(smX.size()[0],-1).sum(dim=1))
			logDetJacob = -(invBitmask*smX).view(smX.size(0),-1).sum(dim=1)
			x = maskedX + invBitmask*((y-self.T(maskedX))/smX.exp())
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


def printBatchNormParameters(bmod):
	print bmod._parameters
	print "mean",bmod.running_mean
	print "var",bmod.running_var
	print "momentum",bmod.momentum
	print "eps",bmod.eps

def batchNormForward(bmod, x):
	prevMean = bmod.running_mean.clone()
	prevVar = bmod.running_var.clone()
	y = bmod(x)

	# Commenting this out because there are numerical stability issues with
	#  using per-batch mean, var to compute inverses during training
	#  Perhaps better to use running mean, var for everything?
	# if bmod.training:
	# 	xmean = (bmod.running_mean-(1-bmod.momentum)*prevMean)/bmod.momentum
	# 	xvar = (bmod.running_var-(1-bmod.momentum)*prevVar)/bmod.momentum
	# 	varSmall = xvar
	# else:
	xmean, xvar = None, None
	varSmall = Variable(bmod.running_var)

	#	printBatchNormParameters(bmod)
	channelSize = x.size(2)*x.size(3)
	weights = bmod.weight # Problem: will this learn in the inverse direction?
	# logDetJacob = -0.5*(weights*(bmod.running_var+bmod.eps)*channelSize).log().sum() #Need the log, I think
	logDetJacob = (weights.abs().log()*channelSize).sum()+(-0.5)*((varSmall+bmod.eps).log()*channelSize).sum()
	logDetJacob = logDetJacob.expand(x.size(0)) #Variable(torch.zeros(x.size(0)).fill_(logDetJacob)) # Same determinant for everything in batch
	return y, logDetJacob, xmean, xvar

# Mean, var are tensors of size y.size(1) if present
def batchNormInvert(bmod, y, mean=None, var=None):
	weights = bmod.weight
	bias = bmod.bias
	nc = y.size(1)
	channelSize = y.size(2)*y.size(3)

	# if mean is None:
	mean = Variable(bmod.running_mean.view(1,nc,1,1).expand_as(y))
	var = Variable(bmod.running_var.view(1,nc,1,1).expand_as(y))
	varSmall = Variable(bmod.running_var)
	# else:
	# 	mean = Variable(mean.view(1,nc,1,1).expand_as(y))
	# 	varSmall = var
	# 	var = Variable(var.view(1,nc,1,1).expand_as(y))

	x = (y-bias.view(1,nc,1,1).expand_as(y))/weights.view(1,nc,1,1).expand_as(y)
	x = x*torch.sqrt(var+bmod.eps)+mean
	# logDetJacob = 0.5*(weights.data*(varSmall+bmod.eps)*channelSize).log().sum()
	logDetJacob = -(weights.abs().log()*channelSize).sum()+(0.5)*((varSmall+bmod.eps).log()*channelSize).sum()

	# Same determinant for everything in batch
	logDetJacob = logDetJacob.expand(x.size(0)) #Variable(torch.zeros(x.size(0)).fill_(logDetJacob))
	return x, logDetJacob


class StageType1(nn.Module):
	def __init__(self, imsize, nc, nh, ks, batchsize):
		super(StageType1, self).__init__()
		assert(imsize%2 == 0)

		self.batchsize = batchsize # Not necessary to run the model, but used to optimize forward calls
		self.imsize = imsize
		self.nc = nc
		self.nh = nh
		self.ks = ks # kernel size

		# Later: can move the index initializations into here for speedup
		#  Induces a dependence on batchsize

		#****** Cached forward squeeze indices
		colperm = torch.cat([torch.arange(0,imsize,2),torch.arange(1,imsize,2)]).long()
		rowperm = torch.cat([torch.arange(0,imsize,2),torch.arange(1,imsize,2)]).long()
		self.horizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchsize,nc,imsize,1)
		self.verticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batchsize,nc,1,imsize)
		self.horizontalIndex = nn.Parameter(self.horizontalIndex, requires_grad=False) #Variable(self.horizontalIndex)
		self.verticalIndex = nn.Parameter(self.verticalIndex, requires_grad=False) #Variable(self.verticalIndex)

		#******* Cached inverse squeeze indices
		rowperm = torch.zeros(imsize)
		colperm = torch.zeros(imsize)
		rowperm[torch.arange(0,imsize,2).long()] = torch.arange(0,imsize/2)
		rowperm[torch.arange(1,imsize,2).long()] = imsize/2+torch.arange(0,imsize/2)
		colperm[torch.arange(0,imsize,2).long()] = torch.arange(0,imsize/2)
		colperm[torch.arange(1,imsize,2).long()] = imsize/2+torch.arange(0,imsize/2)
		rowperm = rowperm.long()
		colperm = colperm.long()
		self.invHorizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchsize,nc,imsize,1)
		self.invVerticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batchsize,nc,1,imsize)
		self.invHorizontalIndex = nn.Parameter(self.invHorizontalIndex, requires_grad=False) #Variable(self.invHorizontalIndex)
		self.invVerticalIndex = nn.Parameter(self.invVerticalIndex, requires_grad=False) #Variable(self.invVerticalIndex)

		# Cached bitmasks
		self.mask = getBitmask(imsize, nc, batchsize, 0) #Variable(getBitmask(imsize, nc, batchsize, 0))
		self.invMask = 1-self.mask

		self.mask = nn.Parameter(self.mask, requires_grad=False)
		self.invMask = nn.Parameter(self.invMask, requires_grad=False)

		#***** Network *******
		self.c1 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align = 0 )
		self.b1 = nn.BatchNorm2d(nc)
		self.c2 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align = 1 )
		self.b2 = nn.BatchNorm2d(nc)
		self.c3 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align = 0 )
		self.b3 = nn.BatchNorm2d(nc)
		# Should I double the hidden layers here too?
		self.c4 = Coupling( S(2*nc, 2*nh, ks), T(2*nc, 2*nh, ks), channelWise = 1 )
		self.b4 = nn.BatchNorm2d(4*nc)
		self.c5 = Coupling( S(2*nc, 2*nh, ks), T(2*nc, 2*nh, ks), channelWise = 0 )
		self.b5 = nn.BatchNorm2d(4*nc)
		self.c6 = Coupling( S(2*nc, 2*nh, ks), T(2*nc, 2*nh, ks), channelWise = 1 )
		self.b6 = nn.BatchNorm2d(4*nc)

		# ***** Cached batchnorm means / values
		self.b1m, self.b1v, self.b2m, self.b2v, self.b3m,\
		 	self.b3v, self.b4m, self.b4v, self.b5m, self.b5v, self.b6m, self.b6v = \
		 	  None, None, None, None, None, None, None, None, None, None, None, None

	# def cuda(self, device=None):
	# 	self.mask = self.mask.cuda()
	# 	self.invMask = self.invMask.cuda()
	# 	self.invHorizontalIndex = self.invHorizontalIndex.cuda()
	# 	self.invVerticalIndex = self.invVerticalIndex.cuda()
	# 	self.horizontalIndex = self.horizontalIndex.cuda()
	# 	self.verticalIndex = self.verticalIndex.cuda()
	# 	return self._apply(lambda t: t.cuda(device))

	def forward(self, x):
		# Size is batchsize x nc x imsize x imsize
		if x.size(0) == self.batchsize:
			bitmask = self.mask
			invBitmask = self.invMask
		else:
			bitmask = Variable(getBitmask(self.imsize, self.nc, x.size(0),0))
			invBitmask = 1-bitmask
			if x.is_cuda:
				bitmask = bitmask.cuda()
				invBitmask = invBitmask.cuda()

		out = x

		if self.training:
			out, det1 = self.c1(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb1, self.b1m, self.b1v = batchNormForward(self.b1,out)
			out, det2 = self.c2(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb2, self.b2m, self.b2v = batchNormForward(self.b2,out)
			out, det3 = self.c3(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb3, self.b3m, self.b3v = batchNormForward(self.b3,out)
			if x.size(0)==self.batchsize:
				out = nvpSqueeze(out, horizontalIndex=self.horizontalIndex, verticalIndex=self.verticalIndex)
			else:
				out = nvpSqueeze(out)
			out, det4 = self.c4(out)
			out, detb4, self.b4m, self.b4v = batchNormForward(self.b4,out)
			out, det5 = self.c5(out)
			out, detb5, self.b5m, self.b5v = batchNormForward(self.b5,out)
			out, det6 = self.c6(out)
			out, detb6, self.b6m, self.b6v = batchNormForward(self.b6,out)
		else:
			out, det1 = self.c1(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb1, _, _ = batchNormForward(self.b1,out)
			out, det2 = self.c2(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb2, _, _ = batchNormForward(self.b2,out)
			out, det3 = self.c3(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb3, _, _ = batchNormForward(self.b3,out)
			if x.size(0)==self.batchsize:
				out = nvpSqueeze(out, horizontalIndex=self.horizontalIndex, verticalIndex=self.verticalIndex)
			else:
				out = nvpSqueeze(out)
			out, det4 = self.c4(out)
			out, detb4, _, _ = batchNormForward(self.b4,out)
			out, det5 = self.c5(out)
			out, detb5, _, _ = batchNormForward(self.b5,out)
			out, det6 = self.c6(out)
			out, detb6, _, _ = batchNormForward(self.b6,out)


		logDetJacob = det1+det2+det3+det4+det5+det6+detb1+detb2+detb3+detb4+detb5+detb6

		# Out size is batchsize x (4 x nc) x (imsize/2) x (imsize/2)
		return out, logDetJacob

	def invert(self,y):
		# Size is batchsize x nc x imsize x imsize
		if y.size(0) == self.batchsize:
			bitmask = self.mask
			invBitmask = self.invMask
		else:
			bitmask = Variable(getBitmask(self.imsize, self.nc, y.size(0),0))
			invBitmask = 1-bitmask
			if y.is_cuda:
				bitmask = bitmask.cuda()
				invBitmask = invBitmask.cuda()

		out = y

		# Consider making this a loop
		if self.training:
			out, detb6 = batchNormInvert(self.b6,out,self.b6m,self.b6v)
			out, det6 = self.c6.invert(out)
			out, detb5 = batchNormInvert(self.b5,out,self.b5m,self.b5v)
			out, det5 = self.c5.invert(out)
			out, detb4 = batchNormInvert(self.b4,out,self.b4m,self.b4v)
			out, det4 = self.c4.invert(out)
			if y.size(0) == self.batchsize:
				out = nvpUnsqueeze(out, horizontalIndex=self.invHorizontalIndex, verticalIndex=self.invVerticalIndex)
			else:
				out = nvpUnsqueeze(out)
			out, detb3 = batchNormInvert(self.b3,out,self.b3m,self.b3v)
			out, det3 = self.c3.invert(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb2 = batchNormInvert(self.b2,out,self.b2m,self.b2v)
			out, det2 = self.c2.invert(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb1 = batchNormInvert(self.b1,out,self.b1m,self.b1v)
			out, det1 = self.c1.invert(out, bitmask=bitmask, invBitmask=invBitmask)
		else:
			out, detb6 = batchNormInvert(self.b6,out)
			out, det6 = self.c6.invert(out)
			out, detb5 = batchNormInvert(self.b5,out)
			out, det5 = self.c5.invert(out)
			out, detb4 = batchNormInvert(self.b4,out)
			out, det4 = self.c4.invert(out)
			if y.size(0) == self.batchsize:
				out = nvpUnsqueeze(out, horizontalIndex=self.invHorizontalIndex, verticalIndex=self.invVerticalIndex)
			else:
				out = nvpUnsqueeze(out)
			out, detb3 = batchNormInvert(self.b3,out)
			out, det3 = self.c3.invert(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb2 = batchNormInvert(self.b2,out)
			out, det2 = self.c2.invert(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb1 = batchNormInvert(self.b1,out)
			out, det1 = self.c1.invert(out, bitmask=bitmask, invBitmask=invBitmask)


		logDetJacob = det1+det2+det3+det4+det5+det6+detb1+detb2+detb3+detb4+detb5+detb6
		return out, logDetJacob


class StageType2(nn.Module):
	def __init__(self, imsize, nc, nh, ks, batchsize):
		super(StageType2, self).__init__()
		self.batchsize = batchsize
		self.imsize = imsize
		self.nc = nc
		self.nh = nh
		self.ks = ks #for convenience below

		self.mask = getBitmask(imsize, nc, batchsize, 0)
		self.invMask = 1-self.mask

		self.mask = nn.Parameter(self.mask, requires_grad=False)
		self.invMask = nn.Parameter(self.invMask, requires_grad=False)

		self.c1 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align=0 )
		self.b1 = nn.BatchNorm2d(nc)
		self.c2 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align=1 )
		self.b2 = nn.BatchNorm2d(nc)
		self.c3 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align=0 )
		self.b3 = nn.BatchNorm2d(nc)
		self.c4 = Coupling( S(nc, nh, ks), T(nc, nh, ks), align=1 ) # ?? Double-check redundancy

		self.b1m, self.b1v, self.b2m, self.b2v, self.b3m, self.b3v = \
			None, None, None, None, None, None

	def forward(self, x):
		# Size is batchsize x nc x imsize x imsize
		if x.size(0) == self.batchsize:
			bitmask = self.mask
			invBitmask = self.invMask
		else:
			bitmask = Variable(getBitmask(self.imsize, self.nc, x.size(0),0))
			invBitmask = 1-bitmask
			if x.is_cuda:
				bitmask = bitmask.cuda()
				invBitmask = invBitmask.cuda()

		out = x

		if self.training:
			out, det1 = self.c1(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb1, self.b1m, self.b1v = batchNormForward(self.b1,out)
			out, det2 = self.c2(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb2, self.b2m, self.b2v = batchNormForward(self.b2,out)
			out, det3 = self.c3(out, bitmask=bitmask, invBitmask=invBitmask)
			out,detb3, self.b3m, self.b3v = batchNormForward(self.b3,out)
			out, det4 = self.c4(out, bitmask=invBitmask, invBitmask=bitmask)
		else:
			out, det1 = self.c1(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb1, _, _ = batchNormForward(self.b1,out)
			out, det2 = self.c2(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb2, _, _ = batchNormForward(self.b2,out)
			out, det3 = self.c3(out, bitmask=bitmask, invBitmask=invBitmask)
			out,detb3, _, _ = batchNormForward(self.b3,out)
			out, det4 = self.c4(out, bitmask=invBitmask, invBitmask=bitmask)


		logDetJacob = det1+det2+det3+det4+detb1+detb2+detb3

		# Out size is batchsize x (4 x nc) x (imsize/2) x (imsize/2)
		return out, logDetJacob

	def invert(self,y):
		if y.size(0) == self.batchsize:
			bitmask = self.mask
			invBitmask = self.invMask
		else:
			bitmask = Variable(getBitmask(self.imsize, self.nc, y.size(0),0))
			invBitmask = 1-bitmask
			if y.is_cuda:
				bitmask = bitmask.cuda()
				invBitmask = invBitmask.cuda()

		out = y

		if self.training:
			out, det4 = self.c4.invert(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb3 = batchNormInvert(self.b3,out, self.b3m, self.b3v)
			out, det3 = self.c3.invert(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb2 = batchNormInvert(self.b2,out, self.b2m, self.b2v)
			out, det2 = self.c2.invert(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb1 = batchNormInvert(self.b1,out, self.b1m, self.b1v)
			out, det1 = self.c1.invert(out, bitmask=bitmask, invBitmask=invBitmask)
		else:
			out, det4 = self.c4.invert(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb3 = batchNormInvert(self.b3,out)
			out, det3 = self.c3.invert(out, bitmask=bitmask, invBitmask=invBitmask)
			out, detb2 = batchNormInvert(self.b2,out)
			out, det2 = self.c2.invert(out, bitmask=invBitmask, invBitmask=bitmask)
			out, detb1 = batchNormInvert(self.b1,out)
			out, det1 = self.c1.invert(out, bitmask=bitmask, invBitmask=invBitmask)


		logDetJacob = det1+det2+det3+det4+detb1+detb2+detb3
		return out, logDetJacob



class RealNVPbase(nn.Module):
	# Future note: use **args as input for every network in the future
	def __init__(self, imsize, nc, nh=64, ks=3, batchsize=64):
		super(RealNVPbase, self).__init__()

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
		self.stage3 = StageType2(imsize/4, 4*nc, 4*nh, ks, batchsize) # nc will overtake nh unless nh starts high enough

	def forward(self, x, invert=False):
		if invert:
			return self.invert(x)

		out = x
		out, det1 = self.stage1(out)
		z1z2, out = torch.chunk(out,2,dim=1)
		out, det2 = self.stage2(out)
		z3, out = torch.chunk(out,2,dim=1)
		z4, det3 = self.stage3(out)

		logDetJacob = det1+det2+det3
		b = x.size(0)
		z = torch.cat((z1z2.contiguous().view(b,-1), z3.contiguous().view(b,-1), z4.contiguous().view(b,-1)), dim=1)

		return z, logDetJacob

	def invert(self, z):
		z1z2 = z[:,:self.totalSize/2]
		# The numbers are wrong; pls fix
		z1z2 = z1z2.contiguous().view(z.size(0), 2*self.nc, self.imsize/2, self.imsize/2)
		z3 = z[:,self.totalSize/2:3*self.totalSize/4]
		z3 = z3.contiguous().view(z.size(0), 4*self.nc, self.imsize/4, self.imsize/4)
		z4 = z[:,3*self.totalSize/4:]
		z4 = z4.contiguous().view(z.size(0), 4*self.nc, self.imsize/4, self.imsize/4)

		z4in, det3 = self.stage3.invert(z4)
		z3 = torch.cat((z3,z4in),dim=1)
		z3in, det2 = self.stage2.invert(z3)
		z1z2 = torch.cat((z1z2, z3in), dim=1)
		x, det1 = self.stage1.invert(z1z2)

		logDetJacob = det1+det2+det3
		return x, logDetJacob

class RealNVP(nn.Module):
	def __init__(self, imsize, nc, nh=64, ks=3, batchsize=64, ngpu=0):
		super(RealNVP, self).__init__()
		self.main = RealNVPbase(imsize, nc, nh, ks, batchsize)
		self.ngpu = ngpu

	def forward(self, x, invert=False):
		if invert:
			return self.invert(x)

		if isinstance(x, torch.cuda.FloatTensor) and self.ngpu > 1:
			y, detx = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
		else:
			y, detx = self.main(x)
		return y, detx

	def invert(self, y):
		if isinstance(y, torch.cuda.FloatTensor) and self.ngpu > 1:
			x, dety = nn.parallel.data_parallel(self.main, y, range(self.ngpu), module_kwargs={'invert':True})
		else:
			x, dety = self.main(y, invert=True)
		return x, dety
		



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

def testRealNVP():
	x = Variable(torch.zeros(2,2,8,8).normal_(), requires_grad=True)
	imsize = 8
	nc = 2
	nh = 8
	ks = 3
	batchsize = 2

	rmod = RealNVP(imsize, nc, nh, ks, batchsize)
	rmod.apply(weights_init)
	rmod.eval()

	y, detforward = rmod(x)
	yinv, detbackward = rmod.invert(y)

	print detbackward.size()
	print y.size()

	y2 = (detbackward*y.sum(dim=1)).sum()
	y2.backward()
	print y2
	print "grad", x.grad.data

#	print x, y, yinv, detforward, detbackward
	print (x-yinv).norm()

def testStage1():
	x = Variable(torch.zeros(2,2,6,6).normal_())
	imsize = 6
	nc = 2
	nh = 8
	ks = 3
	batchsize = 2

	s1mod = StageType1(imsize, nc, nh, ks, batchsize)
	s1mod.apply(weights_init)
	s1mod.eval()
	y, detforward = s1mod(x)
	yinv, detbackward = s1mod.invert(y)

	print x
	print y
	print yinv
	print detforward, detbackward

	print (x-yinv).norm()

def testStage2():
	x = Variable(torch.zeros(2,2,6,6).normal_())
	imsize = 6
	nc = 2
	nh = 8
	ks = 3
	batchsize = 2

	s2mod = StageType2(imsize, nc, nh, ks, batchsize)
	s2mod.apply(weights_init)
	s2mod.eval()
	y, detforward = s2mod(x)
	yinv, detbackward = s2mod.invert(y)

	print x
	print y
	print yinv
	print detforward, detbackward

	print (x-yinv).norm()

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

def testBatchNorm():
	training = Variable(10*torch.zeros(2,2,6,6).normal_()+50)
	btch = nn.BatchNorm2d(2)
	#btch.train() #There is a problem with doing batchnorm during training, possibly...
	#	btch(training)
	btch.eval()	
	#	a = Variable(torch.arange(0,144).resize_((2,2,6,6)))
	a = Variable(torch.zeros(2,2,6,6).normal_())
	#meanEmp = torch.Tensor([a.data[:,i,:,:].mean() for i in range(a.size(1))])
	#varEmp = torch.Tensor([a.data[:,i,:,:].var() for i in range(a.size(1))])
	#print meanEmp, varEmp
	#	mean, var = None, None

	aout, detaout, mean, var = batchNormForward(btch, a)
	ainv, detavin = batchNormInvert(btch, aout, mean, var)

	print detaout, detavin

	#	print a, aout, ainv, detaout, detavin
	print mean, var
	print (a-ainv).norm()

def testCoupling():
	mask = Variable(getBitmask(6,3,2,0))
	c = Coupling( S(3, 5), T(3,5) )
	c.apply(weights_init)

	a = torch.zeros(2,3,6,6)
	#	a.normal_()
	a = Variable(a)
	y, detJacob = c(a, bitmask=mask, invBitmask=1-mask)
	yinv, detJacobInv = c.invert(y, bitmask=mask, invBitmask=1-mask)

	print a, yinv
	print detJacob, detJacobInv
	print (a-yinv).norm()
	print (detJacobInv+detJacob).norm()

def testCouplingChannelWise():
	#	mask = Variable(getBitmask(6,3,2,0))
	c = Coupling( S(2, 5), T(2,5), channelWise=0)
	c.apply(weights_init)

	a = torch.zeros(2,4,6,6)
	a.normal_()
	a = Variable(a)
	y, detJacob = c(a)#, bitmask=mask, invBitmask=1-mask)
	yinv, detJacobInv = c.invert(y)#, bitmask=mask, invBitmask=1-mask)

	print a, yinv, y
	print detJacob, detJacobInv
	print (a-yinv).norm()
	print (detJacobInv+detJacob).norm()



if __name__=='__main__':
#	test(variable=True)
#	testBitmask()
#	testCoupling()
#	testBatchNorm()
#	testCouplingChannelWise()
#	testStage1()
#	testStage2()
	testRealNVP()
