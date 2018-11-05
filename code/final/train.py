# Generic train
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import argparse
from os.path import join
from tensorboardX import SummaryWriter

from code.final.models import getModels, weights_init
from code.final.loaders import getLoaders


# **************** Mode switching ****************
def setTrain(model):
	if isinstance(model, tuple):
		for m in model:
			m.train()
	else:
		m.train()

def setEval(model):
	if isinstance(model, tuple):
		for m in model:
			m.eval()
	else:
		m.eval()


# **************** Tracker objects ****************
class Tracker(object):
	def __init__(self, prefix=''):#, modelType):
		# self.modelType = modelType
		self.prefix = prefix
		self.trainBatchesPath = join(prefix, 'trainCurveBatches')
		self.trainCurvePath = join(prefix, 'trainCurve')
		self.validationCurvePath = join(prefix, 'validationCurve')
		self.imagePath = join(prefix,'Image')

		self.writer = SummaryWriter()
		self.valEpoch = 0
		self.trainEpoch = 0
		self.validationAverage = 0.0
		self.validationIters = 0
		self.trainIters = 0
		self.trainAverage = 0.0

	def trainUpdate(epoch, batchnum, loss):
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

	def validationUpdate(epoch, batchnum, loss):
		if epoch > self.valEpoch:
			self.writer.add_scalar(self.validationCurvePath, self.validationAverage/self.validationIters, epoch)
			self.validationIters = 1
			self.validationAverage = loss
			self.valEpoch = epoch
		else:
			#self.writer.add_scalar('validationCurve', loss, epoch*batchnum)
			self.validationAverage += loss
			self.validationIters += 1

	def addImages(epoch, ims):
		if ims is not None:
			imGrid = vutils.make_grid(ims, normalize=True, scale_each=True)
			writer.add_image(self.imagePath, imGrid, epoch)
		

# **************** Criterion functions ****************
class GANCriterion(nn.Module):
	def __init__(self):
		super(GANCriterion, self).__init__()
		self.lossFunc = nn.BCELoss()

	def forward(target, actual):
		return self.lossFunc(target, actual)


class RegressorCriterion(nn.Module):
	def __init__(self):
		super(RegressorCriterion, self).__init__()

class EmbeddingCriterion(nn.Module):
	def __init__(self):
		super(EmbeddingCriterion, self).__init__()


def getCriterion(criterion):
	if criterion == 'gan':
		return GANCriterion()
	elif criterion == 'wgan':
		pass
	elif criterion == 'flowgan':
		pass
	elif criterion == 'regressor':
		return RegressorCriterion()
	elif criterion == 'embedding':
		return EmbeddingCriterion()



# **************** Model initialization ****************
# Randomly seed the whole process
def randomSeedAll(seed):
	pass

# Apply weights_init to model
def initModel(model):
	if isinstance(model, tuple):
		for m in model:
			m.apply(weights_init)
	else:
		model.apply(weights_init)

# Load a model given a list of file names
def loadModel(model, locations):
	if isinstance(model, tuple):
		for m, l in model,locations:
			m.load_state_dict(torch.load(l))
	else:
		m.load_state_dict(torch.load(locations))


# **************** Model Checkpointing ****************
# Use this as a lambda function with certain locations
def checkpoint(model, locations, epoch=None):
	append = '.pth'
	if epoch is not None:
		append = '_'+str(epoch)+'.pth'

	if isinstance(model, tuple):
		for m, l in model,locations:
			torch.save(m,l+append)
	else:
		torch.save(model, locations+append)




# **************** Optimizers ****************
# Get optimizers for the given model
def getOptimizers(model, lr=0.0002, beta1=0.5, beta2=0.999):
	if isinstance(model, tuple):
		optimizers = []
		for mod in model:
			optimizers.append(torch.optim.Adam(filter(lambda m: m.requires_grad, mod.parameters()), lr=lr, betas=(beta1,beta2)))
		return tuple(optimizers)
	else:
		return torch.optim.Adam(filter(lambda m: m.requires_grad, model.parameters()), lr=lr, betas=(beta1,beta2))





# **************** TrainStep functions ****************
def GANTrainStep(model, batch, optimizers, criterion, cuda):
	netG, netD = model
	optimG, optimD = optimizers

	batch = Variable(batch)
	onesLabel = Variable(torch.ones(batch.size(0)))
	zerosLabel = Variable(torch.zeros(batch.size(0)))
	if cuda:
		onesLabel = onesLabel.cuda()
		zerosLabel = zerosLabel.cuda()
		batch = batch.cuda()

	# ??
	zVals = Variable(torch.Tensor(batch.size(0), netG.nz).normal_(0,1))
	if cuda:
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

	return errG.data[0], errD.data[0]



def supervisedTrainStep(model, batch, optimizer, criterion, cuda):
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

	return err.data[0]






def RegressorTrainStep(model, batch, optimizers, criterion):
	pass

def RegressorValidationStep(model, batch, criterion):
	pass

def EmbeddingTrainStep(model, batch, optimizers, criterion):
	pass

def EmbeddingValidationStep(model, batch, criterion):
	pass

def getTrainFunc(trainfunc, validation = False):
	if trainfunc == 'gan':
		return GANTrainStep
	elif trainfunc == 'flowgan':
		pass
	elif trainfunc == 'regressor':
		if validation:
			return RegressorTrainStep, RegressorValidationStep
		else:
			return RegressorTrainStep
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
	for i in range(startEpoch, epochs):
		imsave=None
		for j, batch in enumerate(loader):
			losses, ims = trainStep(model, batch, optimizers, criterion, cuda)
			if isinstance(trackers, tuple):
				for track, loss in trackers, losses:
					track.trainUpdate(epoch, j, loss)
			else:
				tracker.trainUpdate(epoch, j, losses)
			if j == len(loader)-1:
				imsave = ims #return none if no ims
		if isinstance(trackers, tuple):
			trackers[0].addImages(imsave)
		else:
			trackers.addImages(imsave)
		if i>0 and i%checkpointEvery == 0:
			checkpoint(model, checkpointLocs, i)
		if validationLoader is not None:
			setEval(model)
			for k, valBatch in enumerate(epoch, k, validationLoader):
				valLosses = validationStep(model, valBatch, criterion, cuda)
				if isinstance(trackers, tuple):
					for track, loss in trackers, valLosses:
						track.validationUpdate(epoch, j, loss)
				else:
					tracker.validationUpdate(epoch, j, valLosses)

			setTrain(model)
	setEval(model)
	checkpoint(model, checkpointLocs)


def makeCuda(model):
	if isinstance(model,tuple):
		out = []
		for m in model:
			out.append(m.cuda())
		return tuple(out)
	else:
		return model.cuda()

def makeParallel(model):
	if isinstance(model, tuple):
		newMods = []
		for m in model:
			newMods.append(nn.DataParallel(m))
		return tuple(newMods)
	else:
		return nn.DataParallel(model)




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='dcgan32 | flowgan32 | pixelRegressor32 | deepRegressor32 | embedding32')
	parser.add_argument('--dataset', required=True, help='cifar10 | mnist | lsun | imagenet | folder | lfw | fake')
	parser.add_argument('--dataroot', default=None, help='path to dataset')
	parser.add_argument('--modelroot', help='path to model save location')
	parser.add_argument('--netG', default='', help="path to netG (to continue training)")
	parser.add_argument('--netD', default='', help="path to netD (to continue training)")
	parser.add_argument('--netR', default='', help='Path to regressor')
	parser.add_argument('--netEmb', default='', help='Path to embedding net')
	parser.add_argument('--epochsCompleted', type=int, default=0, help='Number of epochs already completed by loaded models')
	parser.add_argument('--parameterSet', default=None, help='Dict of pre-defined parameters to use as opts')
	parser.add_argument('--supervised', action='store_true', help='Is this a supervised training problem')
	parser.add_argument('--fuzzy', action='store_true', help='Add small random noise to input' )
	parser.add_argument('--validation', action='store_true', help='Use validation set during training')
	parser.add_argument('--trainValProportion', type=float, help='Proportion to split as training data for training/validation')
	parser.add_argument('--deep', action='store_true', help='Using deep features for training')
	parser.add_argument('--criterion', type=str, default='gan', help='Loss criterion for the gan')
	parser.add_argument('--trainFunc', type=str, default='gan', help='The training function to use')


	parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
	parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
	parser.add_argument('--nc', type=int, default=3, 'Colors in image')
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
	# parser.add_argument('--proportions',type=str, help='Probabilities of each class in mnist',default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')

	opt = parser.parse_args()

	# randomSeedAll
	# Print options
	model = getModels(opt.model, nc=opt.nc, imsize=opt.imageSize, hidden=opt.hidden, nz=opt.nz)

	# Init or load model here?

	criterion = getCriterion(opt.criterion)

	haveCuda = torch.cuda.is_available()
	ngpus = torch.cuda.device_count()
	if haveCuda:
		model = makeCuda(model)
		criterion = makeCuda(criterion)
	if ngpus > 1:
		model = makeParallel(model)

	optimizers = getOptimizers(model, lr=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
	loader = getLoaders(loader=opt.dataset, nc=opt.nc, size=opt.imageSize, root=opt.dataroot, batchsize=opt.batchSize, returnLabel=opt.supervised,
	     fuzzy=opt.fuzzy, mode='train', validation=opt.validation, trProp=opt.trainValProportion, deep=opt.deep)
	if opt.validation:
		loader, valLoader = loader
	else:
		valLoader = None


	# tracker = Tracker(...)
	trainStep = getTrainFunc(opt.trainFunc, validation=opt.validation)
	if opt.validation:
		trainStep, valStep = trainStep
	else:
		valStep = None
	# checkpoint = lambda checkpoint(...)

	if 'gan' in opt.trainFunc:
		checkpointLocs = (join(opt.modelroot, 'netG'), join(opt.modelroot, 'netD'))
	elif 'regressor' in opt.trainfunc:
		checkpointLocs = join(opt.modelroot, 'netR')
	elif 'embedding' in opt.trainfunc:
		checkpointLocs = join(opt.modelroot, 'netEmb')

	# initModel or loadModel

#	train(...)
	# startEpoch=opt.epochsCompleted



if __name__=='__main__':
	main()