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
#import pickle
import json

from models import getModels, weights_init
from loaders import getLoaders

writer = SummaryWriter('tensorboard')
now = datetime.datetime.today()
nowStr = now.strftime('%Y_%b_%d_%I_%M_%S')


def saveOpts(dirname, optsDict):
	json.dump(optsDict, open(join(dirname,'opts.json'),'w'))



# **************** Mode switching ****************
def setTrain(model):
	for m in model:
		m.train()

def setEval(model):
	for m in model:
		m.eval()


# **************** Tracker objects ****************
class Tracker(object):
	def __init__(self, prefix='', verbose=True):#, modelType):
		# self.modelType = modelType
		self.prefix = prefix
		self.trainBatchesPath = join(nowStr, prefix, 'trainCurveBatches')
		self.trainCurvePath = join(nowStr, prefix, 'trainCurve')
		self.validationCurvePath = join(nowStr, prefix, 'validationCurve')
		self.imagePath = join(nowStr, prefix,'Image')
		self.verbose = verbose

		self.writer = writer #SummaryWriter('tensorboard')
		self.valEpoch = 0
		self.trainEpoch = 0
		self.validationAverage = 0.0
		self.validationIters = 0
		self.trainIters = 0
		self.trainAverage = 0.0

	def trainUpdate(self, epoch, batchnum, loss):
		if self.verbose and batchnum%50 == 0:
			print "epoch {0} batch {1} {2} loss = {3}".format(epoch, batchnum, self.prefix, loss)
		self.writer.add_scalar(self.trainBatchesPath, loss, epoch*batchnum)
		if epoch > self.trainEpoch:
			self.writer.add_scalar(self.trainCurvePath, self.trainAverage/self.trainIters, epoch)
			self.trainIters = 1
			self.trainAverage = loss
			self.trainEpoch = epoch
		else:
			#self.writer.add_scalar('validationCurve', loss, epoch*batchnum)
			self.trainAverage += loss
			self.trainIters += 1

	def validationUpdate(self, epoch, batchnum, loss):
		if epoch > self.valEpoch:
			self.writer.add_scalar(self.validationCurvePath, self.validationAverage/self.validationIters, epoch)
			self.validationIters = 1
			self.validationAverage = loss
			self.valEpoch = epoch
			if self.verbose:
				print "validation epoch {0} batch {1} {2} loss = {3}".format(epoch, batchnum, self.prefix, loss)
		else:
			#self.writer.add_scalar('validationCurve', loss, epoch*batchnum)
			self.validationAverage += loss
			self.validationIters += 1

	def addImages(self, epoch, ims):
		if self.verbose:
			print "Adding images epoch {0}".format(epoch)
		if ims is not None:
			imGrid = vutils.make_grid(ims, normalize=True, scale_each=True)
			self.writer.add_image(self.imagePath, imGrid, epoch)
		

# **************** Criterion functions ****************
class GANCriterion(nn.Module):
	def __init__(self):
		super(GANCriterion, self).__init__()
		self.lossFunc = nn.BCELoss()

	def forward(self, actual, target):
		return self.lossFunc(actual, target)

class SoftmaxBCE(nn.Module):
	def __init__(self):
		super(SoftmaxBCE, self).__init__()
		self.softmax = nn.Softmax()
		self.bce = nn.BCELoss()

	def forward(self, actual, target):
		actual = self.softmax(actual)
		return self.bce(actual,target)

class RegressorCriterion(nn.Module):
	def __init__(self):
		super(RegressorCriterion, self).__init__()

class EmbeddingCriterion(nn.Module):
	def __init__(self):
		super(EmbeddingCriterion, self).__init__()


def getCriterion(criterion):
	if criterion == 'gan':
		return [GANCriterion()]
	elif criterion == 'bce':
		return [nn.BCELoss()]
	elif criterion == 'softmaxbce':
		return [SoftmaxBCE()]
	elif criterion == 'wgan':
		pass
	elif criterion == 'flowgan':
		pass
	elif criterion == 'l2':
		return [nn.MSELoss()]
	elif criterion == 'embedding':
		return [EmbeddingCriterion()]



# **************** Model initialization ****************
# Randomly seed the whole process
def randomSeedAll(seed):
	pass

# Apply weights_init to model
def initModel(model):
	for m in model:
		m.apply(weights_init)

# Load a model given a list of file names
def loadModel(model, locations):
#	for m, l in model,locations:
	for i, l in enumerate(locations):
		m = model[i]
		if l is not None:
			m.load_state_dict(torch.load(l))


# **************** Model Checkpointing ****************
# Use this as a lambda function with certain locations
def checkpoint(model, locations, epoch=None):
	append = '.pth'
	if epoch is not None:
		append = '_'+str(epoch)+'.pth'

	for i, m in enumerate(model):
		l = locations[i]
		torch.save(m,l+append)




# **************** Optimizers ****************
# Get optimizers for the given model
def getOptimizers(model, lr=0.0002, beta1=0.5, beta2=0.999):
	optimizers = []
	for mod in model:
		optimizers.append(torch.optim.Adam(filter(lambda m: m.requires_grad, mod.parameters()), lr=lr, betas=(beta1,beta2)))
	return optimizers





# **************** TrainStep functions ****************
def GANTrainStep(model, batch, optimizers, criterion, cuda):
	netG, netD = model
	optimG, optimD = optimizers
	criterion = criterion[0]

	nz = netG.numLatent()

	batch = Variable(batch)
	onesLabel = Variable(torch.ones(batch.size(0)))
	zerosLabel = Variable(torch.zeros(batch.size(0)))
	zVals = Variable(torch.Tensor(batch.size(0), nz).normal_(0,1))
	if cuda:
		onesLabel = onesLabel.cuda()
		zerosLabel = zerosLabel.cuda()
		batch = batch.cuda()
		zVals = zVals.cuda()

	netD.zero_grad()
	fake = netG(zVals)
	fakePred = netD(fake.detach())
	realPred = netD(batch)

	errD = criterion(fakePred, zerosLabel)+criterion(realPred,onesLabel)
	errD.backward()
	optimD.step()

	netG.zero_grad()
	gPred = netD(fake)
	errG = criterion(gPred, onesLabel)
	errG.backward()
	optimG.step()

	return (errG.data[0], errD.data[0]), fake.data



def supervisedTrainStep(model, batch, optimizer, criterion, cuda):
	model = model[0]
	optimizer = optimizer[0]
	criterion = criterion[0]
	x, y = batch
	x = Variable(x)
	y = Variable(y)
	if cuda:
		x = x.cuda()
		y = y.cuda()

	model.zero_grad()
	ypred = model(x)
	err = criterion(ypred, y)
	err.backward()
	optimizer.step()

	return [err.data[0]], None

def supervisedValidationStep(model, batch, criterion, cuda):
	model = model[0]
	criterion = criterion[0]
	x, y = batch
	x = Variable(x)
	y = Variable(y)
	if cuda:
		x = x.cuda()
		y = y.cuda()

	ypred = model(x)
	err = criterion(ypred, y)

	return [err.data[0]]




def RegressorTrainStep(model, batch, optimizers, criterion):
	pass

def RegressorValidationStep(model, batch, criterion):
	pass

def EmbeddingTrainStep(model, batch, optimizer, criterion, cuda):
	model = model[0]
	optimizer = optimizer[0]
	criterion = criterion[0]

	x, y = batch
	y = y.view(-1,1) # make 2d
	label = torch.zeros(y.size(0), model.numOutClasses()) #10 for now
	label.scatter_(1,y,1)
	y = label

	x = Variable(x)
	y = Variable(y)
	if cuda:
		x = x.cuda()
		y = y.cuda()

	model.zero_grad()
	ypred = model(x)
	err = criterion(ypred, y)
	err.backward()
	optimizer.step()

	return [err.data[0]], None
 

def EmbeddingValidationStep(model, batch, criterion, cuda):
	model = model[0]
	criterion = criterion[0]

	x, y = batch
	y = y.view(-1,1) # make 2d
	label = torch.zeros(y.size(0), model.numOutClasses) #10 for now
	label.scatter_(1,y,1)
	y = label

	x = Variable(x)
	y = Variable(y)
	if cuda:
		x = x.cuda()
		y = y.cuda()

	ypred = model(x)
	err = criterion(ypred, y)

	return [err.data[0]]

def getTrainFunc(trainfunc, validation = False):
	if trainfunc == 'gan':
		return GANTrainStep
	elif trainfunc == 'flowgan':
		pass
	elif trainfunc == 'regressor':
		if validation:
			return supervisedTrainStep, supervisedValidationStep
		else:
			return supervisedTrainStep
	elif trainfunc == 'embedding':
		if validation:
			return EmbeddingTrainStep, EmbeddingValidationStep
		else:
			return EmbeddingTrainStep


# model: nn.Module or tuple of nn.Modules to be trained
# trainStep(model, batch, optimizers, criterion)  :  a function that runs one training batch
#    and returns the average loss on the batch
# optimizers: a torch.nn.optim module, or tuple of such modules, one per network in the model
# criterion: an nn.Module that computes the loss for the model
# tracker: something to track network progress. Has as update function
# checkpoint(model) : a function that saves the model
# options: a dictionary of training options
def train(model, trainStep, optimizers, loader, criterion, trackers, checkpointLocs, cuda, epochs=10, 
	startEpoch=0, checkpointEvery=2, validationLoader=None, validationStep=None):
	setTrain(model)
	for epoch in range(startEpoch, epochs):
		imsave=None
		for j, batch in enumerate(loader):
			losses, ims = trainStep(model, batch, optimizers, criterion, cuda)
			for t, track in enumerate(trackers):
				loss = losses[t]
				track.trainUpdate(epoch, j, loss)
			if j == len(loader)-1:
				imsave = ims #return none if no ims
		trackers[0].addImages(epoch, imsave)
		if epoch>0 and epoch%checkpointEvery == 0:
			checkpoint(model, checkpointLocs, epoch)
		if validationLoader is not None:
			setEval(model)
			for k, valBatch in enumerate(validationLoader):
				valLosses = validationStep(model, valBatch, criterion, cuda)
				for t, track in enumerate(trackers):
					loss = valLosses[t]
					track.validationUpdate(epoch, k, loss)
			setTrain(model)
	setEval(model)
	checkpoint(model, checkpointLocs, epochs)


def makeCuda(model):
	out = []
	for m in model:
		out.append(m.cuda())
	return out

def makeParallel(model):
	newMods = []
	for m in model:
		newMods.append(nn.DataParallel(m))
	return newMods




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='dcgan', help='dcgan | flowgan | pixelRegressor | deepRegressor | embedding')
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
	parser.add_argument('--trainValProportion', default=0.9, type=float, help='Proportion to split as training data for training/validation')
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
	parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
	parser.add_argument('--checkpointEvery', type=int, default=2, help='Checkpoint after every n epochs')
	parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
	# parser.add_argument('--cuda', action='store_true', help='enables cuda')
	# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
	# parser.add_argument('--outf', default='/fs/vulcan-scratch/krusinga/distGAN/checkpoints',
	#                     help='folder to output images and model checkpoints')
	parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
	# parser.add_argument('--proportions',type=str, help='Probabilities of each class in mnist',default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')

	opt = parser.parse_args()

	saveOpts(opt.modelroot, opt.__dict__)

	# randomSeedAll
	# Print options

	# ********* Getting model **********
	model = getModels(opt.model, nc=opt.nc, imsize=opt.imageSize, hidden=opt.hidden, nz=opt.nz)

	# Init or load model here?

	# ********* Getting criterion *********
	criterion = getCriterion(opt.criterion)

	# ********** Checking cuda ************
	haveCuda = torch.cuda.is_available()
	ngpus = torch.cuda.device_count()
	if haveCuda:
		model = makeCuda(model)
		criterion = makeCuda(criterion)
#		criterion = criterion.cuda()
	if ngpus > 1:
		model = makeParallel(model)

	# *********** Getting optimizers **********
	optimizers = getOptimizers(model, lr=opt.lr, beta1=opt.beta1, beta2=opt.beta2)

	# *********** Getting loader **************
	loader = getLoaders(loader=opt.dataset, nc=opt.nc, size=opt.imageSize, root=opt.dataroot, batchsize=opt.batchSize, returnLabel=opt.supervised,
	     fuzzy=opt.fuzzy, mode='train', validation=opt.validation, trProp=opt.trainValProportion, deep=opt.deep)
	if opt.validation:
		loader, valLoader = loader
	else:
		valLoader = None


	# *********** Getting training step function ***********
	trainStep = getTrainFunc(opt.trainFunc, validation=opt.validation)
	if opt.validation:
		trainStep, valStep = trainStep
	else:
		valStep = None

	# *********** Getting checkpointing info, file loading info, tracking info ***********
	if 'gan' in opt.trainFunc:
		checkpointLocs = (join(opt.modelroot, 'netG'), join(opt.modelroot, 'netD'))
		loaderLocs = (opt.netG, opt.netD)
		trackers = (Tracker('netG'), Tracker('netD'))
	elif 'regressor' in opt.trainFunc:
		checkpointLocs = [join(opt.modelroot, 'netR')]
		loaderLocs = [opt.netR]
		trackers = [Tracker('Regressor')]
	elif 'embedding' in opt.trainFunc:
		checkpointLocs = [join(opt.modelroot, 'netEmb')]
		loaderLocs = [opt.netEmb]
		trackers = [Tracker('Embedding')]

	# *********** Initialize and/or load models ***************
	initModel(model)
	loadModel(model, loaderLocs)


	# ************** Train the model ****************
	train(model, trainStep, optimizers, loader, criterion, trackers, checkpointLocs, haveCuda, epochs=opt.epochs, 
		startEpoch=opt.epochsCompleted, checkpointEvery=opt.checkpointEvery, validationLoader=valLoader, validationStep=valStep)



if __name__=='__main__':
	main()
