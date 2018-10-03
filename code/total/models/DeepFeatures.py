# Deep features for training
from mlworkflow import Data
from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init, Lenet28, Lenet32, Lenet64, Lenet128

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np 
import torch.optim.lr_scheduler as lr_scheduler


class LenetModel(ModelTemplate):
  def __init__(self, config, args):
    super(LenetModel,self).__init__(config, args)
    self.opt = {
      'nc':3,
      'imSize':28,
      'nOutClasses':10, #Number of classes to predict
      'ngpu':0, # This depends on cuda availability, fix later
      'cuda':False,
      'lr':0.0002,
      'beta1':0.5,
      'lenetClass':Lenet28,
      'lenetKey':'lenet',
      'lenetInstance':-1,
      'lenetExpNum':-1,
      'checkpointEvery':5
    }
    args = copy(args)
    self.opt.update(args)

    self.log(str(self.opt))

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.inShape = (self.nc, self.imSize, self.imSize)
    self.log("inShape is "+str(self.inShape))


    self.lenet = self.lenetClass(ngpu=self.ngpu, nc=self.nc)
    self.criterion = nn.BCELoss()
    self.optimizer = optim.Adam(self.lenet.parameters(), lr=self.lr, betas=(self.beta1,0.999))
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)


    ######
    if (self.lenetInstance == -1):
      self.lenetInstance = self.getLatestInstance(self.lenetKey, self.lenetExpNum)

    if (self.lenetInstance is not None):
      self.log("Loading Lenet from instance {0}".format(self.lenetInstance))
      self.lenet.load_state_dict(self.load(self.lenetKey, instance=self.lenetInstance, number=self.lenetExpNum, loader='torch'))

      if self.checkExists('lenetState', instance=self.lenetInstance, number=self.lenetExpNum):
        self.lossCurve, self.errorCurve = self.load('lenetState', instance=self.lenetInstance, number=self.lenetExpNum, loader='pickle')
      else:
        self.lossCurve = ([], [], [])
        self.errorCurve = ([], [], [])
    else:
      self.log("Training lenet from scratch")
      self.lenetInstance = -1
      self.lenet.apply(weights_init)
      self.lossCurve = ([], [], [])
      self.errorCurve = ([], [], [])


    if self.cuda:
      self.lenet = self.lenet.cuda()
      self.criterion = self.criterion.cuda()



  def train(self, loaderTemplate, nepochs):
    # Reset for this run
    dataloaderTrain = loaderTemplate.getDataloader(outShape=self.inShape, mode='train', returnLabel=True)
    dataloaderTest = loaderTemplate.getDataloader(outShape=self.inShape, mode='test', returnLabel=True)

    startEpoch = self.lenetInstance+1
    for epoch in range(startEpoch, nepochs):
      self.log("===Begin epoch {0}".format(epoch))
      self.scheduler.step()
      self.trainEpoch(dataloaderTrain, epoch)
      self.testEpoch(dataloaderTest, epoch)
      self.lossCurve[0].append(epoch)
      self.errorCurve[0].append(epoch)
    # Save checkpoint
      if (epoch+1)%self.checkpointEvery == 0:
        self.saveCheckpoint(checkpointNum = epoch)
      
    self.saveCheckpoint(checkpointNum=(nepochs-1))


  def trainEpoch(self, dataloader, epoch):
    self.lenet.train()
    losses = AverageMeter()
    abserror = AverageMeter()

    for i, (data,y) in enumerate(dataloader):
      self.log("Iteration {0}".format(i))
      
      self.lenet.zero_grad()
      batchSize = data.size(0)
      y = y.view(-1,1) # make 2d
      label = torch.zeros(batchSize, self.nOutClasses) #10 for now
      label.scatter_(1,y,1) 

      ##### Stopped here for Monday

      if self.cuda:
        data = data.cuda()
        label = label.cuda()

      data = Variable(data)
      label = Variable(label)

      _, output = self.lenet(data)
      err = self.criterion(output, label)
      losses.update(err.data[0], data.size(0))
      abserror.update((output.data-label.data).abs_().mean(), data.size(0))

      err.backward()
      self.optimizer.step()

    self.log("Epoch {0}, loss={1}, abserror={2}".format(epoch, losses.avg, abserror.avg))
    self.lossCurve[1].append(losses.avg)
    self.errorCurve[1].append(abserror.avg)

      
  # Sample 
  def sample(self, inputs):
    self.lenet.eval()
    if self.cuda:
      inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = self.lenet(inputs)
    return outputs.data.cpu()

  # method should be one of 'numerical', 'exact'
  def probSample(self, nSamples, deepFeatures=None, method='numerical', epsilon=1e-5):
    pass

  # Move this to a library function later
  # Get the probabilities of a set of codes
  def getProbs(self, codes, method='numerical', epsilon=1e-5):
    pass
      

  # Get the generator codes that reconstruct the images most closely
  def nearestInput(self, ims):
    pass

  def testEpoch(self, dataloader, epoch):
    losses = AverageMeter()
    abserror = AverageMeter()
    self.lenet.eval()
    for i, (data, y) in enumerate(dataloader):
      batchSize = data.size(0)
      y = y.view(-1,1)
      label = torch.zeros(batchSize, self.nOutClasses)
      label.scatter_(1,y,1) 

      if self.cuda:
        data = data.cuda()
        label = label.cuda()
      data = Variable(data, volatile=True)
      label = Variable(label, volatile=True)

      _, output = self.lenet(data)#, 5)
      err = self.criterion(output, label)

      losses.update(err.data[0], data.size(0))
      abserror.update((output.data - label.data).abs_().mean(), data.size(0))

    self.log('Testing epoch %d, loss = %d, abserror=%d'%(epoch,losses.avg,abserror.avg))
    self.lossCurve[2].append(losses.avg)
    self.errorCurve[2].append(abserror.avg)

  def saveCheckpoint(self, checkpointNum=None):
    if checkpointNum is not None:
      self.save(self.lenet.state_dict(), 'lenet', instance=checkpointNum, saver='torch')
      self.save((self.lossCurve, self.errorCurve), 'lenetState', instance=checkpointNum, saver='pickle')
    else:
      self.save(self.lenet.state_dict(), 'lenet', saver='torch')
      self.save((self.lossCurve, self.errorCurve), 'lenetState', saver='pickle')


  def getAnalysisData(self):
    train = {
      'data':np.array(self.lossCurve),
      'legend':['Train Loss','Test Loss'],
      'legendLoc':'upper right',
      'xlabel':'Epoch',
      'ylabel':'Loss',
      'title':'Train/Test loss curve',
      'format':'png'
    }
    test = {
      'data':np.array(self.errorCurve),
      'legend':['Train Error','Test Error'],
      'legendLoc':'upper right',
      'xlabel':'Epoch',
      'ylabel':'Error',
      'title':'Train/Test error curve',
      'format':'png'
    }
    lossDat = Data(train, 'lineplot', 'lossCurve')
    errorDat = Data(test, 'lineplot', 'errorCurve')
    return [lossDat, errorDat]


class LenetSize28Cols3(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 3
		args['imSize'] = 28
		args['lenetClass'] = Lenet28
		super(LenetSize28Cols3, self).__init__(config, args)

class LenetSize28Cols1(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 1
		args['imSize'] = 28
		args['lenetClass'] = Lenet28
		super(LenetSize28Cols1, self).__init__(config, args)

class LenetSize32Cols3(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 3
		args['imSize'] = 32
		args['lenetClass'] = Lenet32
		super(LenetSize32Cols3, self).__init__(config, args)

class LenetSize32Cols1(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 1
		args['imSize'] = 32
		args['lenetClass'] = Lenet32
		super(LenetSize32Cols1, self).__init__(config, args)

class LenetSize64Cols3(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 3
		args['imSize'] = 64
		args['lenetClass'] = Lenet64
		super(LenetSize64Cols3, self).__init__(config, args)

class LenetSize64Cols1(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 1
		args['imSize'] = 64
		args['lenetClass'] = Lenet64
		super(LenetSize64Cols1, self).__init__(config, args)

class LenetSize128Cols3(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 3
		args['imSize'] = 128
		args['lenetClass'] = Lenet128
		super(LenetSize128Cols3, self).__init__(config, args)

class LenetSize128Cols1(LenetModel):
	def __init__(self, config, args):
		args = copy(args)
		args['nc'] = 1
		args['imSize'] = 128
		args['lenetClass'] = Lenet128
		super(LenetSize128Cols1, self).__init__(config, args)
