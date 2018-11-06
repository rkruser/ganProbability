# Generic train
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import argparse
from os.path import join
from tensorboardX import SummaryWriter
import sys
import datetime

from models import getModels, weights_init
from loaders import getLoaders

import json


def loadOpts(dirname):
	return json.load(open(join(dirname,'opts.json'),'r'))

# Types of sampling
# Deep features
# Regressor probabilities on some data
# GAN probabilities
#   numerical
#   backprop
#   z optim
#   BiGAN



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='dcgan32', help='dcgan32 | flowgan32 | pixelRegressor32 | deepRegressor32 | embedding32')
	parser.add_argument('--dataset', default='mnist', help='cifar10 | mnist | lsun | imagenet | folder | lfw | fake')
	parser.add_argument('--dataroot', default=None, help='path to dataset')
	parser.add_argument('--modelroot', default='generated/final/dcgan_mnist', help='path to model save location')
	parser.add_argument('--netG', type=str, default=None, help="path to netG (to continue training)")
	parser.add_argument('--netD', type=str, default=None, help="path to netD (to continue training)")
	parser.add_argument('--netR', type=str, default=None, help='Path to regressor')
	parser.add_argument('--netEmb', type=str, default=None, help='Path to embedding net')
	parser.add_argument('--epochsCompleted', type=int, default=0, help='Number of epochs already completed by loaded models')
	parser.add_argument('--parameterSet', default=None, help='Dict of pre-defined parameters to use as opts')
	parser.add_argument('--supervised', action='store_true', help='Is this a supervised training problem')
	parser.add_argument('--fuzzy', action='store_true', help='Add small random noise to input' )
	parser.add_argument('--validation', action='store_true', help='Use validation set during training')
	parser.add_argument('--trainValProportion', default=0.8, type=float, help='Proportion to split as training data for training/validation')
	parser.add_argument('--deep', action='store_true', help='Using deep features for training')
	parser.add_argument('--criterion', type=str, default='gan', help='Loss criterion for the gan')
	parser.add_argument('--trainFunc', type=str, default='gan', help='The training function to use')


	parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
	parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
	parser.add_argument('--nc', type=int, default=3, help='Colors in image')
	parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
	parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
	# parser.add_argument('--gfeats', type=int, default=64, help='Hidden features in G net')
	# parser.add_argument('--dfeats', type=int, default=64, help='Hidden features in D net')
	# parser.add_argument('--rfeats', type=int, default=64, help='Hidden features in regressor net')
	# parser.add_argument('--efeats', type=int, default=64, help='Hidden features in embedding net')
	parser.add_argument('--hidden', type=int, default=128, help='Hidden features in networks')
	# parser.add_argument('--classindex', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
	parser.add_argument('--checkpointEvery', type=int, default=5, help='Checkpoint after every n epochs')
	parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
	# parser.add_argument('--cuda', action='store_true', help='enables cuda')
	# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
	# parser.add_argument('--outf', default='/fs/vulcan-scratch/krusinga/distGAN/checkpoints',
	#                     help='folder to output images and model checkpoints')
	parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

	parser.add_argument('--useSavedOpts', action='store_true', help='load from saved opts json')
	# parser.add_argument('--proportions',type=str, help='Probabilities of each class in mnist',default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')

	opt = parser.parse_args()

	if opt.useSavedOpts:
		opt.__dict__.update(loadOpts(opt.modelroot))



	# ********* Getting model **********
	model = getModels(opt.model, nc=opt.nc, imsize=opt.imageSize, hidden=opt.hidden, nz=opt.nz)

	# Init or load model here?



if __name__=='__main__':
	main()
