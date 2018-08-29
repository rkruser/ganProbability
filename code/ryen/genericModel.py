import torch
import torch.nn as nn
import numpy as np
from code.ryen.utility import AverageMeter

class DeepModel(object):
  def __init__(self):
    self.networks = {}
    self.analysis = {}
    self.logfunc = None
    self.checkpointFuncs = {}
    self.checkpointEvery = 1
#    self.errGmeter = AverageMeter()
#    self.errDmeter = AverageMeter()
#    self.errGlist = []
#    self.errDlist = []
#    self.images = []

  def registerLogger(self, logger):
    self.logfunc = logger # A function to call for logging

  # Each func must take a keyword, an instance, and the data
  def registerCheckpointing(self, checkpointFuncs, interval):
    self.checkpointFuncs = checkpointFuncs
    self.checkpointEvery = interval

  # func(keyword, instance, data)
  def checkpoint(self, epoch, nEpochs):
    if instance%self.checkpointEvery == 0 or instance == nEpochs-1:
      if len(self.checkpointFuncs) > 0:
        for netkey in self.networks:
          if netkey in self.checkpointFuncs:
            self.checkpointFuncs[netkey](netkey, instance, self.networks[netkey])
        for akey in self.analysis:
          if akey in self.checkpointFuncs:
            self.checkpointFuncs[akey](akey, instance, self.analysis[akey])

  def log(self, msg):
    if self.logfunc is not None:
      self.logfunc(msg)

  # nEpochs is integer
  # dataset is torch dataloader or similar
  def train(self, nEpochs, dataset):
    self.preTrain()
    for i in range(nEpochs):
      self.log("===Begin training epoch %d"%i)
      self.preEpoch(i)
      for j, data in enumerate(dataset):
        self.log("Iter %d"%j)
        self.preLoop(j)
        self.innerLoop(i, j, len(dataset), data)
        self.postLoop(j)
      self.postEpoch(i)
      self.checkpoint(i, nEpochs)
    self.postTrain()

  def preTrain(self):
    for k in self.networks:
      self.networks[k].train()

  def postTrain(self):
    pass

  def preEpoch(self, i):
    pass

  def postEpoch(self, i):
    pass # Do checkpointing or something

  def preLoop(self, j):
    pass

  def postLoop(self, j):
    pass

  # Main updates
  def innerLoop(self, epoch, j, size, data):
    pass


class genericGAN(DeepModel):
  def __init__(self, netG, netD, nz, InitializerFunc, optimizerG, optimizerD, LossCriterion, ngpu=0):
    super(genericGAN, self).__init__()
    self.nz = nz
    self.netG = netG
    self.netD = netD
    self.networks = {'netG':self.netG, 'netD':self.netD}
    self.initFunc = InitializerFunc
    self.optimG = optimizerG
    self.optimD = optimizerD
    self.criterion = LossCriterion
    self.ngpu = ngpu

    # Recording
    self.errGmeter = AverageMeter()
    self.errDmeter = AverageMeter()
    self.analysis['images'] = []
    self.analysis['gLosses'] = []
    self.analysis['dLosses'] = []

  def innerLoop(self, epoch, j, size, data):
    # Discriminator update
    self.netD.zero_grad()
    batchSize = data.size(0)
    labelsReal = torch.Tensor(batchSize).fill_(1.0)
    labelsFake = torch.Tensor(batchSize).fill_(0.0)
    zCodes = torch.Tensor(batchSize, self.nz, 1, 1).normal_(0,1)
    if self.ngpu > 0:
      data = data.cuda()
      zCodes = zCodes.cuda()
      labelsReal = labelsReal.cuda()
      labelsFake = labelsFake.cuda()
    data = Variable(data)
    zCodes = Variable(zCodes)
    labelsReal = Variable(labelsReal)
    labelsFake = Variable(labelsFake)

    # Real D outputs
    dRealOut = self.netD(data)
    errDreal = self.criterion(dRealOut, errD)
    errDreal.backward()

    # Fake D outputs
    fakeIms = self.netG(zCodes)
    dFakeOut = self.netD(fakeIms.detach())
    errDfake = self.criterion(dFakeOut, labelsFake)
    errDfake.backward()
    errD = errDreal + errDfake
    self.optimizerD.step()

    # Train G net
    self.netG.zero_grad()
    dgFakeOut = self.netD(fakeIms)
    errG = self.criterion(dgFakeOut, labelsReal)
    errG.backward()
    self.optimizerG.step()
     
    # Recording results
    self.recordLosses(errG.data[0], batchSize, errD.data[0], batchSize)
    if j == (size-2):
      self.recordImages(fakeIms.data.cpu())


  def recordLosses(self, gErr, gBatchSize, dErr, dBatchSize):
    self.errGmeter.update(gErr, gBatchSize)
    self.errDmeter.update(dErr, dBatchSize)

  def postEpoch(self, i):
    self.analysis['gLosses'].append(self.errGmeter.avg)
    self.analysis['dLosses'].append(self.errDmeter.avg)
    self.errGmeter.reset()
    self.errDmeter.reset()

  def recordImages(self, imTensor):
    self.analysis['images'].append(np.array((np.transpose(imTensor,(0,2,3,1))[:16]*0.5+0.5)*255,dtype='uint8'))
 


