from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init, NetP28, NetP32, NetP64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler


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


class RegressorModel(ModelTemplate):
  def __init__(self, config, args):
    super(Regressor,self).__init__(config, args)
    self.opt = {
      'nz': 100,
      'npf': 64,
      'nc':3,
      'imSize':28,
      'ngpu':0,
      'cuda':False,
      'lr':0.0002,
      'beta1':0.5,
      'netPclass':NetP28,
      'netPkey':'',
      'netPinstance':-1,
      'netPexpNum':-1,
      'checkpointEvery':10
    }
    args = copy(args)
    self.opt.update(args)

    self.log(str(self.opt))

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.log("inShape is "+str(self.outShape))

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


    self.netP = self.netPclass(ngpu=self.ngpu, nc=self.nc, npf=self.npf)
    self.criterion = nn.SmoothL1Loss(size_average=True)
    self.optimizerP = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1,0.999))
    self.scheduler = lr_scheduler.StepLR(self.optimizerP, step_size=200, gamma=0.1)

    # Load netG, newly or from file
    if self.netPkey != '':
      self.netP.load_state_dict(self.load(self.netPkey, instance=self.netGinstance, number=self.netGexpNum))
    else:
      self.netP.apply(weights_init)

    if self.cuda:
      self.netP.cuda()
      self.criterion.cuda()


    self.lossCurve = ([], [], [])
    self.errorCurve = ([], [], [])


  def train(self, loaderTemplate, nepochs):
    # Reset for this run
    dataloaderTrain = loaderTemplate.getDataloader(outShape=self.outShape, mode='train', returnClass=False)
    dataloaderTest = loaderTemplate.getDataloader(outShape=self.outShape, mode='test', returnClass=False)

    for epoch in range(nepochs):
      self.log("===Begin epoch {0}".format(epoch))
      scheduler.step()
      self.trainEpoch(dataLoaderTrain)
      self.testEpoch(dataLoaderTest)
      self.lossCurve[0].append(epoch)
      self.errorCurve[0].append(epoch)
    # Save checkpoint
      if (epoch+1)%self.checkpointEvery == 0:
        self.save(checkpointNum = epoch)
      
    self.save()


  def trainEpoch(self, dataloader):
    self.netP.train()
    losses = AverageMeter()
    abserror = AverageMeter()

    for i, (data,label) in enumerate(dataloader):
      self.log("Iteration {0}".format(i))
      
      self.netP.zero_grad()
      batchSize = data.size(0)

      if self.cuda:
        data.cuda()
        label.cuda()

      data = Variable(data)
      label = Variable(label)

      output = self.netP(data)
      err = self.criterion(output, label)
      losses.update(err.data[0], data.size(0))
      abserror.update((output.data-label).abs_().mean(), data.size(0))

      err.backward()
      self.optimizerP.step()

    self.log("Epoch {0}, loss={1}, abserror={2}".format(epoch, losses.avg, abserror.avg))
    self.lossCurve[1].append(losses.avg)
    self.errorCurve[1].append(abserror.avg)

      
  # Sample 
  def sample(self, inputs):
    outProbs = self.netP(inputs)
    return outProbs

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

  def testEpoch(self, dataloader):
    losses = AverageMeter()
    abserror = AverageMeter()
    self.netP.eval()
    for i, (data, label) in enumerate(dataloader):
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        data = Variable(data, volatile=True)
        label = Variable(label, volatile=True)

        output = self.netP(data)#, 5)
        err = criterion(output, label)

        losses.update(err.data[0], data.size(0))
        abserror.update((output.data - label).abs_().mean(), data.size(0))

#    self.log('Testing epoch %d, loss = %d, abserror=%d'%(epoch,losses.avg,abserror.avg))
#      self.testCurve[0].append(epoch)
    self.lossCurve[2].append(losses.avg)
    self.errorCurve[2].append(abserror.avg)

  def saveCheckpoint(self, checkpointNum=None):
    if checkpointNum is not None:
      self.save(self.netP.state_dict(), 'netP', instance=checkpointNum, saver='torch')
    else:
      self.save(self.netP.state_dict(), 'netP', saver='torch')


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

