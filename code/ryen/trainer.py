from mlworkflow import Operator, Data
from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from code.ryen.utility import AverageMeter

class TrainerRyen(Operator):
  def __init__(self, config, args):
    super(TrainerRyen, self).__init__(config, args)
    opt = {
      'nepochs':1
    }
    opt.update(args)
    self.opt = edict(opt)
    assert(len(self.dependencies) == 2)
    self.dataLoader = self.dependencies[0]
    self.modelLoader = self.dependencies[1]

    # Recording
    self.errG = []
    self.errD = []
    self.images = []

  def run(self):
    data = self.dataLoader.getLoader()
    model = self.modelLoader.getModel()

    self.train(model, data)

    # Maybe add testing later


  # Need:
  # netG, netD, cuda, criterion, nz, optimizerG, optimizerD

  # No longer need Variable class I think
  # in the new version of torch
  def train(self, model, dataloader):
    self.log("Begin training")
    # Place networks in training mode
    model.netG.train()
    model.netD.train()
    for epoch in range(self.opt.nepochs):
      self.log("===Begin epoch %d"%epoch)
      gLosses = AverageMeter()
      dLosses = AverageMeter()
      for i, data in enumerate(dataloader):
        # Train the D portion on real ims
        self.log("Iteration %d"%i)
        model.netD.zero_grad()
        if isinstance(data, torch.Tensor):
          dataX = data # split into data and label? Nope
        else:
          dataX, _ = data
        batch_size = dataX.size(0)
        labelsReal = torch.Tensor(batch_size).fill_(1.0)
        labelsFake = torch.Tensor(batch_size).fill_(0.0)
        zCodes = torch.Tensor(batch_size, model.nz,1,1).normal_(0,1)
        if model.cuda:
          dataX = dataX.cuda()
          zCodes = zCodes.cuda()
          labelsReal = labelsReal.cuda()
          labelsFake = labelsFake.cuda()

        dataX = Variable(dataX)
        zCodes = Variable(zCodes)
        labelsReal = Variable(labelsReal)
        labelsFake = Variable(labelsFake)

        dReal, _ = model.netD(dataX)
        errDreal = model.criterion(dReal, labelsReal)
        errDreal.backward()
        
        # Train D using fake labels
        fakeIms = model.netG(zCodes)
        dFake, dRecon = model.netD(fakeIms.detach()) # detach??
        errDfake = model.criterion(dFake, labelsFake)+model.reconScale*model.reconstructionLoss(dRecon, zCodes.view(batch_size, model.nz))
        errDfake.backward()
        errD = errDreal + errDfake
        model.optimizerD.step()
        
        # Train G network        

        model.netG.zero_grad()
#        labelsForged = torch.Tensor(batch_size).fill_(1.0)
        gFake, gRecon = model.netD(fakeIms) # no detach this time
        errG = model.criterion(gFake, labelsReal) - model.reconScale*model.reconstructionLoss(gRecon, zCodes.view(batch_size, model.nz)) # Question: should the recon loss be propagated back to G here?
        errG.backward()
        model.optimizerG.step()

        # Info
        gLosses.update(errG.data[0],batch_size)
        dLosses.update(errD.data[0],batch_size)
        if i==(len(dataloader)-2):
          self.images.append(np.array((np.transpose(fakeIms.data.cpu().numpy(),(0,2,3,1))[:16]*0.5+0.5)*255,dtype='uint8'))

    
      self.errG.append(gLosses.avg)
      self.errD.append(dLosses.avg)

     
    self.log("Saving netG")
    self.files.save(model.netG.state_dict(), 'netG', saver='torch')
    self.log("Saving netD")
    self.files.save(model.netD.state_dict(), 'netD', saver='torch')  
    self.log("Saving analysis data")
    self.files.save((self.errG, self.errD, self.images), 'analysisPickle', saver='pickle')
  

  def getAnalysisData(self):
    assert(len(self.errG) == len(self.errD))
    errInfo = {
      'data':np.array([range(len(self.errG)),self.errG,self.errD]),
      'legend':['Generator error','Discriminator Error'],
      'xlabel':'Epoch',
      'ylabel':'Average error',
      'title':'GAN training curve',
      'format':'png'
    }
    errAnalysis = Data(errInfo, 'lineplot', 'ganTrainPlot')
    results = [errAnalysis]
    for i, im in enumerate(self.images):
      results.append(Data({'images':im,'dpi':400}, 'imageArray', 'ganSampleIms', instance=i))
    return results


