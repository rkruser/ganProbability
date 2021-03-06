from copy import copy
from mlworkflow import Operator, Data

from code.total.models.nnModels import weights_init, NetG28, NetD28, NetG32, NetD32, NetG64, NetD64, NetP28, NetP32, NetP64
from code.total.loaders.MatLoaders import MatLoader

import torch
from torch.autograd import Variable
import numpy as np

# Here is a new comment

class GANTrain(Operator):
  def __init__(self, config, args):
    super(GANTrain, self).__init__(config, args)
    args = copy(args)
    self.opt = {
      'nGANepochs':20
    }
    self.opt.update(args)

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.dataloader = self.dependencies[0]
    self.ganModel = self.dependencies[1]
#		self.regressorModel = self.dependencies[2]
	#	self.sampler = self.dependencies[3]


  def run(self):
    pid = self.getPID()
    if pid == 0:
      self.ganModel.train(self.dataloader, self.nGANepochs)

    # In experiment, do nothing if pid is not 0
    # In Sample object, wait until netG exists
    # Then can do everything in one config, not two


class RegressorTrain(Operator):
  def __init__(self, config, args):
    super(RegressorTrain, self).__init__(config, args)
    args = copy(args)
    self.opt = {
      'nRegressorEpochs':20
    }
    self.opt.update(args)

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.probLoader = self.dependencies[0]
    self.regressorModel = self.dependencies[1]

  def run(self):
    pid = self.getPID()
    if pid == 0:
      self.regressorModel.train(self.probLoader, self.nRegressorEpochs)


class RegressorTest(Operator):
  def __init__(self, config, args):
    super(RegressorTest, self).__init__(config, args)
    args = copy(args)
    self.opt = {
#      'dataset':'mnist28',
      'take': 25,
      'distribution':None,
      'nRegressorSamples':1000,
      'testRegressorDeepFeatures':False,
      'cuda':False
    }
    self.opt.update(args)

    for key in self.opt:
      setattr(self, key, self.opt[key])

    #self.datasetPath = self.getPath(self.dataset)

    self.dataloader = self.dependencies[0]
    self.regressor = self.dependencies[1]
    if self.testRegressorDeepFeatures:
      self.deepModel = self.dependencies[2]

    self.analysisData = []

  def run(self):
#    dataloader = MatLoader(self.datasetPath, distribution=self.distribution, mode='test', returnLabel=True)
    dataloader = self.dataloader.getDataset(outShape=self.regressor.inShape, distribution=self.distribution, mode='test', returnLabel=True)
    # Sample
    imArr = []
    labelArr = []
    for i in range(self.nRegressorSamples):
      im, y = dataloader[i]
      imArr.append(im)
      labelArr.append(y)

    ims = torch.stack(imArr, dim=0)
    if self.testRegressorDeepFeatures:
      deepModel = self.deepModel.getModel()
      if self.cuda:
        varIms = ims.cuda()
      varIms = Variable(varIms)
      _, embeddedIms = deepModel(varIms)
      imProbs = self.regressor.sample(embeddedIms.data)
    else:
      imProbs = self.regressor.sample(ims)

    #######
    sortedInds = np.argsort(imProbs)
    topN = sortedInds[-self.take:]
    bottomN = sortedInds[:self.take]

    imTop = np.array(ims[topN])
  #  imTop = (imTop+1.0)*127.5
  #  imTop = imTop.astype(np.uint8)
    imTop = imTop*0.5+0.5 #Transform changes depending on dataset
    # Later make the transform flexible
#    imTop = np.transpose(imTop,(2,3,1,0)) # Transpose to RGB format
    imTop = np.transpose(imTop,(0,2,3,1))
    # put number of samples in last dimension because
    # that's hows matlab needs it for imshow

    self.analysisData.append(Data({'images':imTop},'imageArray','topDataIms'))

    imBot = np.array(ims[bottomN])
    imBot = imBot*0.5+0.5
    imBot = np.transpose(imBot,(0,2,3,1))

    self.analysisData.append(Data({'images':imBot},'imageArray','botDataIms')) 

    #####

    resultData = {
      'images':np.array(ims),
      'prob':np.array(imProbs),
      'label':np.array(labelArr)
    }

    self.save(resultData, 'regressorResults', saver='mat')


  def getAnalysisData(self):
    return self.analysisData


class DeepFeatureTrain(Operator):
  def __init__(self, config, args):
    super(DeepFeatureTrain, self).__init__(config, args)
    args = copy(args)
    self.opt = {
#      'dataset':'mnist28',
      'nDeepEpochs': 50
    }
    self.opt.update(args)

    for key in self.opt:
      setattr(self, key, self.opt[key])

    #self.datasetPath = self.getPath(self.dataset)

    self.dataloader = self.dependencies[0]
    self.deepmodel = self.dependencies[1]

    self.analysisData = []

  def run(self):
    self.deepmodel.train(self.dataloader, self.nDeepEpochs)


# Actually, there is no need for the following details
# Just need to instantate GANModel operator objects
class Experiment(Operator):
  def __init__(self, config, args):
    super(Experiment, self).__init__(config, args)
    args = copy(args)
    self.opt = {
      'expNetGclass':NetG28,
      'expNetG':'NetG',
      'expNetGinstance':-1,
      'expNetGexpNum':-1,
      'expNetDclass':NetD28,
      'expNetD':'NetD',
      'expNetDinstance':-1,
      'expNetDexpNum':-1,
      'expNetPClass':NetP28,
      'expNetP':'NetP',
      'expNetPinstance':-1,
      'expNetPexpNum':-1
    }
    self.opt.update(args)

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.netG = self.expNetGclass(nz=self.nz, ngf=self.ngf, nc=self.nc, ngpu=self.ngpu)
    self.netD = self.expNetDclass(nz=self.nz, ngf=self.ngf, nc=self.nc, ngpu=self.ngpu)
    self.netP = self.expNetPclass(ngpu=self.ngpu, nc=self.nc, npf=self.npf)

    # Load netG, newly or from file
    if self.netGkey != '':
      self.netG.load_state_dict(self.load(self.expNetGkey, instance=self.expNetGinstance, number=self.expNetGexpNum))
    else:
      self.netG.apply(weights_init)

    # Load netD, newly or from file
    if self.netDkey != '':
      self.netD.load_state_dict(self.load(self.expNetDkey, instance=self.expNetDinstance, number=self.expNetDexpNum))
    else:
      self.netD.apply(weights_init)

    if self.netPkey != '':
      self.netP.load_state_dict(self.load(self.expNetPkey, instance=self.expNetPinstance, number=self.expNetPexpNum))
    else:
      self.netP.apply(weights_init)

    if self.cuda:
      self.netG = self.netG.cuda()
      self.netD = self.netD.cuda()
      self.netP = self.netP.cuda()
  
    
# Can derive from experiment and implement the run function
#  Derived classes can know what their dependencies are




