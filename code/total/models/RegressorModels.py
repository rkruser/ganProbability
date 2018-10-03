from mlworkflow import Data
from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init, NetP28, NetP32, NetP64 #, Lenet28, Lenet32, Lenet64, Lenet128
from code.total.models.nnModels import Lenet28, Lenet32, Lenet64, Lenet128, DeepRegressor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler



class RegressorModel(ModelTemplate):
  def __init__(self, config, args):
    super(RegressorModel,self).__init__(config, args)
    self.opt = {
      'npf': 64,
      'nc':3,
      'imSize':28,
      'ngpu':0,
      'cuda':False,
      'lr':0.0002,
      'beta1':0.5,
      # If using a deep embedding
      'usesEmbeddingModel':False,
      'embeddingModelKey':'lenet',
      'embeddingModelInstance':-1,
      'embeddingModelExpNum':-1,
      'nOutFeatures':500,
      'embeddingModelClass':None,
      # Regular netP
      'netPclass':NetP28,
      'netPkey':'netP',
      'netPinstance':-1,
      'netPexpNum':-1,
      'checkpointEvery':10
    }
    args = copy(args)
    self.opt.update(args)

    self.log(str(self.opt))

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.inShape = (self.nc, self.imSize, self.imSize)

    self.log("inShape is "+str(self.inShape))

    # Perhaps there is a pythonic way to convert dict items to attributes
#    self.nz = self.opt['nz']
#    self.npf = self.opt['npf']
#    self.nc = self.opt['nc']
#    self.imSize = self.opt['imSize']
#    self.lr = self.opt['lr']
#    self.beta1 = self.opt['beta1']
#    self.ngpu = self.opt['ngpu']
#    self.cuda = self.opt['cuda']
#    self.inShape = (self.nc,self.imSize,self.imSize)
#
#
#    self.netPclass = self.opt['netPclass']
#    self.netPkey = self.opt['netP']
#    self.netPinstance = self.opt['netPinstance']
#    self.netPexpNum = self.opt['netPexpNum']
#
#    self.checkpointEvery = self.opt['checkpointEvery']


    if self.usesEmbeddingModel:
      self.log("Regressor is using a deep feature embedding")
      self.embeddingModel = self.embeddingModelClass(self.ngpu, self.nc)
      if self.netPinstance == -1:
        self.netPinstance = self.getLatestInstance(self.netPkey, self.netPexpNum)

      if self.netPinstance is None:
        # If this netP has not been saved before, load the embedding model from a file but nothing else
        self.log("Loading an embedding model from file, training netP from scratch")
        if self.embeddingModelInstance == -1:
          self.embeddingModelInstance = self.getLatestInstance(self.embeddingModelKey, self.embeddingModelExpNum)
        self.embeddingModel.load_state_dict(self.load(self.embeddingModelKey, instance=self.embeddingModelInstance, number=self.embeddingModelExpNum, loader='torch'))
        self.netP = self.netPclass(self.embeddingModel, self.nOutFeatures, self.ngpu)        
        self.netPinstance = -1
#        self.netP.apply(weights_init) #Applied in the init function
        self.lossCurve = ([], [], [])
        self.errorCurve = ([], [], [])
      else:
        # If this netP has been saved before, load the embedding model and eveything else at once
        self.log("Loading deep feature netP instance {0}".format(self.netPinstance))

        self.netP = self.netPclass(self.embeddingModel, self.nOutFeatures, self.ngpu)
        self.netP.load_state_dict(self.load(self.netPkey, instance=self.netPinstance, number=self.netPexpNum, loader='torch'))
        if self.checkExists('regressorState', instance=self.netPinstance, number=self.netPexpNum):
          self.lossCurve, self.errorCurve = self.load('regressorState', instance=self.netPinstance, number=self.netPexpNum, loader='pickle')
        else:
          self.lossCurve = ([], [], [])
          self.errorCurve = ([], [], [])


    else:
      self.netP = self.netPclass(ngpu=self.ngpu, nc=self.nc, npf=self.npf)

      if self.netPinstance == -1:
        self.netPinstance = self.getLatestInstance(self.netPkey, self.netPexpNum)

      # Load netP, newly or from file
      if self.netPinstance is not None:
        self.log("Loading netP instance {0}".format(self.netPinstance))
        self.netP.load_state_dict(self.load(self.netPkey, instance=self.netPinstance, number=self.netPexpNum, loader='torch'))
        if self.checkExists('regressorState', instance=self.netPinstance, number=self.netPexpNum):
          self.lossCurve, self.errorCurve = self.load('regressorState', instance=self.netPinstance, number=self.netPexpNum, loader='pickle')
        else:
          self.lossCurve = ([], [], [])
          self.errorCurve = ([], [], [])
      else:
        self.log("Training netP from scratch")
        self.netPinstance = -1
        self.netP.apply(weights_init)
        self.lossCurve = ([], [], [])
        self.errorCurve = ([], [], [])


    self.log(str(self.netP))

    self.criterion = nn.SmoothL1Loss(size_average=True)
    self.optimizerP = optim.Adam(filter(lambda m: m.requires_grad, self.netP.parameters()), lr=self.lr, betas=(self.beta1,0.999))
    self.scheduler = lr_scheduler.StepLR(self.optimizerP, step_size=200, gamma=0.1)
    if self.cuda:
      self.netP = self.netP.cuda()
      self.criterion = self.criterion.cuda()


    

  def train(self, loaderTemplate, nepochs):
    # Reset for this run
    dataloaderTrain = loaderTemplate.getDataloader(outShape=self.inShape, mode='train')
    dataloaderTest = loaderTemplate.getDataloader(outShape=self.inShape, mode='test')

    startEpoch = self.netPinstance+1
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
    self.netP.train()
    losses = AverageMeter()
    abserror = AverageMeter()

    for i, (data,label) in enumerate(dataloader):
      self.log("Iteration {0}".format(i))
      
      self.netP.zero_grad()
      batchSize = data.size(0)

      if self.cuda:
        data = data.cuda()
        label = label.cuda()

      data = Variable(data)
      label = Variable(label)

      output = self.netP(data)
      err = self.criterion(output, label)
      losses.update(err.data[0], data.size(0))
      abserror.update((output.data-label.data).abs_().mean(), data.size(0))

      err.backward()
      self.optimizerP.step()

    self.log("Epoch {0}, loss={1}, abserror={2}".format(epoch, losses.avg, abserror.avg))
    self.lossCurve[1].append(losses.avg)
    self.errorCurve[1].append(abserror.avg)

      
  # Sample 
  def sample(self, inputs):
    self.netP.eval()
    if self.cuda:
      inputs = inputs.cuda()
    inputs = Variable(inputs)
    outProbs = self.netP(inputs)
    return outProbs.data.cpu()

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
    self.netP.eval()
    for i, (data, label) in enumerate(dataloader):
        if self.cuda:
          data = data.cuda()
          label = label.cuda()
        data = Variable(data, volatile=True)
        label = Variable(label, volatile=True)

        output = self.netP(data)#, 5)
        err = self.criterion(output, label)

        losses.update(err.data[0], data.size(0))
        abserror.update((output.data - label.data).abs_().mean(), data.size(0))

    self.log('Testing epoch %d, loss = %d, abserror=%d'%(epoch,losses.avg,abserror.avg))
    self.lossCurve[2].append(losses.avg)
    self.errorCurve[2].append(abserror.avg)

  def saveCheckpoint(self, checkpointNum=None):
    if checkpointNum is not None:
      self.save(self.netP.state_dict(), 'netP', instance=checkpointNum, saver='torch')
      self.save((self.lossCurve, self.errorCurve), 'regressorState', instance=checkpointNum, saver='pickle')
    else:
      self.save(self.netP.state_dict(), 'netP', saver='torch')
      self.save((self.lossCurve, self.errorCurve), 'regressorState', saver='pickle')


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


class RegressorSize28Col3(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 28
    args['netPclass'] = NetP28
    super(RegressorSize28Col3, self).__init__(config, args)

class RegressorSize28Col1(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 28
    args['netPclass'] = NetP28
    super(RegressorSize28Col3, self).__init__(config, args)

class RegressorSize32Col3(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 32
    args['netPclass'] = NetP32
    super(RegressorSize32Col3, self).__init__(config, args)

class RegressorSize32Col1(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 32
    args['netPclass'] = NetP32
    super(RegressorSize32Col1, self).__init__(config, args)

class RegressorSize64Col3(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 64
    args['netPclass'] = NetP64
    super(RegressorSize64Col3, self).__init__(config, args)

class RegressorSize64Col1(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 64
    args['netPclass'] = NetP64
    super(RegressorSize64Col1, self).__init__(config, args)

########### Deep Models

class DeepRegressorSize28Col3(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 28
    args['netPclass'] = DeepRegressor
    args['usesEmbeddingModel'] = True
    args['embeddingModelClass'] = Lenet28
    args['nOutFeatures'] = 500
    super(DeepRegressorSize28Col3, self).__init__(config, args)

class DeepRegressorSize28Col1(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 28
    args['netPclass'] = DeepRegressor
    args['usesEmbeddingModel'] = True
    args['embeddingModelClass'] = Lenet28
    args['nOutFeatures'] = 500
    super(DeepRegressorSize28Col3, self).__init__(config, args)

class DeepRegressorSize32Col3(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 32
    args['netPclass'] = DeepRegressor
    args['usesEmbeddingModel'] = True
    args['embeddingModelClass'] = Lenet32
    args['nOutFeatures'] = 500
    super(DeepRegressorSize32Col3, self).__init__(config, args)

class DeepRegressorSize32Col1(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 32
    args['netPclass'] = DeepRegressor
    args['usesEmbeddingModel'] = True
    args['embeddingModelClass'] = Lenet32
    args['nOutFeatures'] = 500
    super(DeepRegressorSize32Col1, self).__init__(config, args)

class DeepRegressorSize64Col3(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 64
    args['netPclass'] = DeepRegressor
    args['usesEmbeddingModel'] = True
    args['embeddingModelClass'] = Lenet64
    args['nOutFeatures'] = 500

    super(DeepRegressorSize64Col3, self).__init__(config, args)

class DeepRegressorSize64Col1(RegressorModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 64
    args['netPclass'] = DeepRegressor
    args['usesEmbeddingModel'] = True
    args['embeddingModelClass'] = Lenet64
    args['nOutFeatures'] = 500

    super(DeepRegressorSize64Col1, self).__init__(config, args)

