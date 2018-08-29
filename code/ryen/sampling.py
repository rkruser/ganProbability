from mlworkflow import Operator
from easydict import EasyDict as edict
from copy import copy

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# Want to do the following:
# Sample z, run through generator and discriminator, 
#    get reconstructed code, compare with original
# Sample real ims, run through discriminator, then run code through generator,
#   compare output images to real images
# See if Z codes produced by discriminator are normally distributed (visualize random projections)
# Train regressor with z codes

class SampleAnalyze(Operator):
  def __init__(self, config, args):
    super(SampleAnalyze, self).__init__(config, args)
    #args = copy(args)
    opt = {}
    opt.update(args)
    self.opt = edict(opt)

    self.dataLoader = self.dependencies[0]
    self.modelLoader = self.dependencies[1]

  def sampleReconstruct(self, netG, netD):
    Zsamples = torch.FloatTensor(self.opt.nSamples, self.opt.nz,1,1).normal_(0,1)
    if self.opt.cuda:
      Zsamples = Zsamples.cuda()
    Zsamples = Variable(Zsamples)

    _, recon = netD(netG(Zsamples))

    original = Zsamples.data.cpu().squeeze().numpy()
    reconstructed = recon.data.cpu().squeeze().numpy()
  
    diff = original - reconstructed
    errors = np.sqrt(np.sum(diff*diff,1))

    toSave = {
      'originalZ': original,
      'reconstructedZ': reconstructed,
      'errors': errors
    }

    self.log("Saving sampleReconSave")
    self.files.save(toSave, 'sampleReconSave', saver='mat')

  def realReconstruct(self, netG, netD):
    dataset = self.dataLoader.getDataset()
    #realSamples = dataset[np.random.choice(len(dataset),self.opt.nSamples,replace=False)]
    inds = np.random.choice(len(dataset),self.opt.nSamples,replace=False)
    realSamples = [dataset[inds[k]][0].unsqueeze(0) for k in range(len(inds))]
    realSamples = torch.cat(realSamples, 0)
    if self.opt.cuda:
      realSamples = realSamples.cuda()
    realSamples = Variable(realSamples)

    _, recon = netD(realSamples)
    reconstructedIms = netG(recon.unsqueeze(2).unsqueeze(3))

    realIms = realSamples.data.cpu().numpy()
    reconIms = reconstructedIms.data.cpu().numpy() 
    reconCodes = recon.data.cpu().squeeze().numpy()

    toSave = {
      'realIms': realIms,
      'reconIms': reconIms,
      'reconCodes': reconCodes
    }

    self.log("Saving sampleRealSave")
    self.files.save(toSave, 'sampleRealSave', saver='mat')
    
  def run(self):
    netG = self.modelLoader.getGenerator()
    netD = self.modelLoader.getDiscriminator()
    netG.eval()
    netD.eval()
    if self.opt.cuda:
      netG = netG.cuda()
      netD = netD.cuda()

    self.sampleReconstruct(netG, netD)
    self.realReconstruct(netG, netD)  

    

