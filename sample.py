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
import numpy as np

from models import getModels, weights_init
from loaders import getLoaders
from train import setTrain, setEval, makeCuda, makeParallel, getLoaders, initModel, loadModel
from models import DeepFeaturesWrapper

import json

from scipy.io import loadmat, savemat



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


# data is a tensor with a large batch of data
# network is one neural network

# Presume that data is already a variable, and already cuda if necessary
def runHuge(network, data, nchunks=50):#, cuda=False):
    chunks = torch.chunk(data,nchunks)
    outChunks = []
    for ch in chunks:
        result = network(chVar)
        outChunks.append(result)
    return torch.cat(outChunks)


# dataloader is unused, just a placeholder
def sampleNumericalProbabilities(ganModel, nSamples, eps, dataloader, cuda):
	netG, netD = ganModel

	nz = netG.numLatent()
	outshape = netG.outshape()
	totalOut = netG.numOutDims()

	noise = torch.FloatTensor(2*nz+1,nz)
	b = torch.eye(nz)*eps
	codes = torch.FloatTensor(nSamples,nz).normal_(0,1)

	if cuda:
		noise = noise.cuda()
		b = b.cuda()
		codes = codes.cuda()

	# Using log10 because it's more intuitive
	gauss_const = -nz*np.log10(np.sqrt(2*np.pi))
	log_const = np.log10(np.exp(1))

	images = np.empty([nSamples]+outshape)
	probs = np.empty([nSamples])
	jacob = np.empty([nSamples, min(totalOut,nz)]) 


	for i in range(nSamples):
		J = np.empty([totalOut, nz])

		a = codes[i].view(1,-1)
		noise.copy_(torch.cat((a,a+b,a-b),0))
		noisev = Variable(noise, volatile=True) #volatile helps with memory somehow

		# So as not to run out of memory
		if totalOut <= 3072:
			fakeIms = netG(noisev)
		else:
			fakeIms = runHuge(netG, noisev, nchunks=10)

		images[i] = fakeIms[0,:].data.cpu().numpy().reshape(outshape)
		I = fakeIms.data.cpu().numpy().reshape(2*nz+1,-1)
		J = (I[1:nz+1,:]-I[nz+1:,:]).transpose()/(2*eps)

		R = np.linalg.qr(J, mode='r')
		Z = a.cpu().numpy()
		diag = R.diagonal().copy()
		jacob[i] = diag.copy() # No modification yet
		diag[np.where(np.abs(diag) < 1e-20)] = 1
		probs[i] = gauss_const - log_const*0.5*np.sum(Z**2) - np.log10(np.abs(diag)).sum()


	# Add in codes
	allData = {
				'images': images.astype(np.float32),
				'probs': probs.astype(np.float32),
				'jacob': jacob.astype(np.float32),
				'codes': codes.cpu().numpy()
				}

	return allData


# dataloader and eps are unused, just placeholders
def sampleBackpropProbabilities(ganModel, nSamples, eps, dataloader, cuda):
	netG, netD = ganModel
	nz = netG.numLatent()
	outshape = netG.outshape()
	totalOut = netG.numOutDims()

	noise = torch.FloatTensor(1,nz)
	codes = torch.FloatTensor(nSamples,nz).normal_(0,1)

	if cuda:
		noise = noise.cuda()
		codes = codes.cuda()

	# Using log10 because it's more intuitive
	gauss_const = -nz*np.log10(np.sqrt(2*np.pi))
	log_const = np.log10(np.exp(1))

	images = np.empty([nSamples]+outshape)
	probs = np.empty([nSamples])
	jacob = np.empty([nSamples, min(totalOut,nz)]) 

	for i in range(nSamples):
		J = np.empty([totalOut, nz])

		a = codes[i].view(1,-1)
		noise.copy_(a)
		noisev = Variable(noise, requires_grad = True) #volatile helps with memory somehow

		fakeIms = netG(noisev)
		fake = fakeIms.view(1,-1)

		for k in range(totalOut):
			if k%100 == 0:
				print "  ",k
			netG.zero_grad()
			# noisev.zero_grad() #??
			fake[0,k].backward(retain_graph=True) # ??
			J[k] = noisev.grad.data.cpu().numpy().squeeze()

		images[i] = fakeIms[0,:].data.cpu().numpy().reshape(outshape)
		# I = fake.data.cpu().numpy().reshape(2*nz+1,-1)
		# J = (I[1:nz+1,:]-I[nz+1:,:]).transpose()/(2*eps)

		R = np.linalg.qr(J, mode='r')
		Z = a.cpu().numpy()
		diag = R.diagonal().copy()
		jacob[i] = diag.copy() # No modification yet
		diag[np.where(np.abs(diag) < 1e-20)] = 1
		probs[i] = gauss_const - log_const*0.5*np.sum(Z**2) - np.log10(np.abs(diag)).sum()



	# Add in codes
	allData = {
				'images': images.astype(np.float32),
				'probs': probs.astype(np.float32),
				'jacob': jacob.astype(np.float32),
				'codes': codes.cpu().numpy()
				}

	return allData


#def sampleMog


# Optimize for each z input
# write as needed
def sampleZOptim(ganModel, nSamples, eps, dataloader, cuda):
	pass

# nSamples and eps are unused
# **** Problem: dataloader should not randomize batches
def sampleEmbeddings(embeddingModel, nSamples, eps, dataloader, cuda):
	netEmb = embeddingModel[0]

	chunks = []
	ychunks = []
	for batch in dataloader:
		if isinstance(batch, list):
			x, y = batch
		else:
			x = batch
			y = None
		x = Variable(x)
		if cuda:
			x = x.cuda()

		emb = netEmb(x)
		chunks.append(emb.data)

		if y is not None:
			ychunks.append(y)

	alldata = torch.cat(chunks,dim=0)
	if len(ychunks) > 0:
		allY = torch.cat(ychunks)
		return {'X':alldata.cpu().numpy(), 'Y':allY.numpy()}
	else:
		return {'X':alldata.cpu().numpy()}


# nSamples and eps are unused
def sampleRegressor(regressorModel, nSamples, eps, dataloader, cuda):
	netR = regressorModel[0]

	chunks = []
	for x in dataloader:
		x = Variable(x)
		if cuda:
			x = x.cuda()

		pvals = netR(x)
		chunks.append(pvals.data)


	allprobs = torch.cat(chunks)
	return {'probs':allprobs.cpu().numpy()}








def getSampleFunc(funcName):
	if funcName == 'numerical':
		return sampleNumericalProbabilities
	elif funcName == 'backprop':
		return sampleBackpropProbabilities
	elif funcName == 'embedding':
		return sampleEmbeddings
	elif funcName == 'regressor':
		return sampleRegressor




def sampler(model, sampleFunc, file, cuda=False, nsamples=None, dataloader=None, eps=1e-5):
	setEval(model)
#	if cuda:
#		makeCuda(model)
	samples = sampleFunc(model, nsamples, eps, dataloader, cuda)
	# samples is a torch tensor?

	savemat(file, samples)




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='dcgan', help='dcgan | flowgan | pixelRegressor | deepRegressor | embedding')
	parser.add_argument('--dataset', default=None, help='cifar10 | mnist | lsun | imagenet | folder | lfw | fake')
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
	parser.add_argument('--samplePrefix', default='samples', help='The file to save the samples to')
	parser.add_argument('--sampleFunc', required=True, default='numerical', help='How to sample the model')
	parser.add_argument('--saveDir', required=True, help='Where to save the samples')
	parser.add_argument('--nsamples', type=int, default=None, help='number of samples')
	parser.add_argument('--eps', type=float, default=1e-5, help='Numerical epsilon')
	parser.add_argument('--deepModel', default=None, help='The deep model to append on the end of a generator')
	parser.add_argument('--datamode', type=str, default='train', help='train | test')
	parser.add_argument('--returnEmbeddingFeats', action='store_true', help='For embeddings, return the layer before the last layer, rather than the last layer')


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
	parser.add_argument('--useLargerEmbedding', action='store_true', help='Use the feature embedding right before the final layer')
	# parser.add_argument('--proportions',type=str, help='Probabilities of each class in mnist',default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')

	opt = parser.parse_args()

	if opt.useSavedOpts:
		# something wrong here?
		opt.__dict__.update(loadOpts(opt.modelroot))


	# ******* File locations *********
	if 'gan' in opt.model:
		loaderLocs = (opt.netG, opt.netD)
	elif 'Reg' in opt.model:
		loaderLocs = [opt.netR]
	elif 'Emb' in opt.model or 'emb' in opt.model or 'densenet' in opt.model:
		loaderLocs = [opt.netEmb]
	else:
		loaderLocs = None


	# ********* Getting model **********
	model = getModels(opt.model, nc=opt.nc, imsize=opt.imageSize, hidden=opt.hidden, nz=opt.nz, returnFeats=opt.returnEmbeddingFeats)
	# initModel(model)
	loadModel(model, loaderLocs)

	if opt.useLargerEmbedding:
		model.setArg(0)

	# Append deep features to end
	if opt.deepModel is not None:
		# **** Need to also load the deep model
		deepModel = getModels(opt.deepModel, nc=opt.nc, imsize=opt.imageSize, hidden=opt.hidden, nz=opt.nz)
		loadModel(deepModel, [opt.netEmb])
		model[0] = DeepFeaturesWrapper(model[0], deepModel[0])

	haveCuda = torch.cuda.is_available()
	ngpus = torch.cuda.device_count()
	if haveCuda:
		model = makeCuda(model)

	if ngpus > 1:
		model = makeParallel(model)




	# ********** Get sampler **********
	sampleFunc = getSampleFunc(opt.sampleFunc)


	# ********** Get dataset, if necessary **************
	if opt.dataset is not None:
		dataloader = getLoaders(loader=opt.dataset, nc=opt.nc, size=opt.imageSize, root=opt.dataroot, batchsize=opt.batchSize, returnLabel=opt.supervised,
	     fuzzy=opt.fuzzy, mode=opt.datamode, validation=opt.validation, trProp=opt.trainValProportion, deep=opt.deep, shuffle=False)
	else:
		dataloader = None


	# ********** Run sampler **********
	sampler(model, sampleFunc, join(opt.saveDir, opt.samplePrefix+'.mat'), 
		cuda=haveCuda, nsamples=opt.nsamples, dataloader=dataloader, eps=opt.eps)






if __name__=='__main__':
	main()
