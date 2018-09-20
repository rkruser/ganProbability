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
      'netP':'',
      'netPinstance':-1,
      'netPexpNum':-1,
      'checkpointEvery':10
    }
    args = copy(args)
    self.opt.update(args)

    self.log(str(self.opt))

    # Perhaps there is a pythonic way to convert dict items to attributes
    self.nz = self.opt['nz']
    self.npf = self.opt['npf']
    self.nc = self.opt['nc']
    self.imSize = self.opt['imSize']
    self.lr = self.opt['lr']
    self.beta1 = self.opt['beta1']
    self.ngpu = self.opt['ngpu']
    self.cuda = self.opt['cuda']
    self.inShape = (self.nc,self.imSize,self.imSize)

    self.log("inShape is "+str(self.outShape))

    self.netPclass = self.opt['netPclass']
    self.netPkey = self.opt['netP']
    self.netPinstance = self.opt['netPinstance']
    self.netPexpNum = self.opt['netPexpNum']

    self.checkpointEvery = self.opt['checkpointEvery']


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



  def train(self, loaderTemplate, nepochs):
    # Reset for this run
    self.images = []
    self.errG = []
    self.errD = []

    dataloader = loaderTemplate.getDataloader(outShape=self.outShape, mode='train', returnClass=False)

    for epoch in range(nepochs):
      self.log("===Begin epoch {0}".format(epoch))
      gLosses = AverageMeter()
      dLosses = AverageMeter()
      for i, data in enumerate(dataloader):
        self.log("Iteration {0}".format(i))
        
        self.netD.zero_grad()
        batchSize = data.size(0)
        labelsReal = torch.Tensor(batchSize).fill_(1.0)
        labelsFake = torch.Tensor(batchSize).fill_(0.0)
        zCodesD = torch.Tensor(batchize, self.nz, 1,1).normal_(0,1)
        zCodesG = torch.Tensor(batchize, self.nz, 1,1).normal_(0,1)

        if self.cuda:
          data.cuda()
          zCodesD.cuda()
          zCodesG.cuda()
          labelsReal.cuda()
          labelsFake.cuda()

        data = Variable(data)
        labelsReal = Variable(labelsReal)
        zCodesD = Variable(zCodesD)
        zCodesG = Variable(zCodesG)
        labelsFake = Variable(labelsFake)
        
        # Running through discriminator
        dPredictionsReal = self.netD(data)
        errDreal = self.criterion(dPredictionsReal, labelsReal)
        errDreal.backward()

        fakeImsD = self.netG(zCodesD)
        dPredictionsFake = self.netD(fakeImsD.detach())
        errDfake = self.criterion(dPredictionsFake, labelsFake)
        errDfake.backward()

        errD = errDreal + errDfake
        self.optimizerD.step()
        
        # Running through Generator
          # Do I need to sample this separately?
          # or can I use existing samples?
        self.netG.zero_grad()
        fakeImsG = self.netG(zCodesG)
        gPredictionsFake = self.netD(fakeImsG)
        errG = self.criterion(gPredictionsFake, labelsReal)
        errG.backward()
        self.optimizerG.step()

        # Extract data from run
        gLosses.update(errG.data[0], batchSize)
        dLosses.update(errD.data[0], batchSize)
        self.errG.append(gLosses.avg)
        self.errD.append(dLosses.avg)

        if i == (len(dataloader)-2) and (epoch+1)%self.checkpointEvery == 0:
          self.images.append(np.array((np.transpose(fakeImsG.data.cpu().numpy(),(0,2,3,1))[:16]*0.5+0.5)*255,dtype='uint8'))

      # Save checkpoint
      if (epoch+1)%self.checkpointEvery == 0:
        self.save(checkpointNum = epoch)

    self.save()
      
  # Sample 
  def sample(self, nSamples):
    pass

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

  def test(self, loaderTemplate, nepochs):
    pass



  def saveCheckpoint(self, checkpointNum=None):
    if checkpointNum is not None:
      self.save(self.netG.state_dict(), 'netP', instance=checkpointNum, saver='torch')
    else:
      self.save(self.netG.state_dict(), 'netP', saver='torch')


  def getAnalysisData(self):
    pass

