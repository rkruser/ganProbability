from mlworkflow import Operator, Data
from easydict import EasyDict as edict

import numpy as np
import json
import scipy.io as sio
from scipy.stats import norm
import numpy.random as nrand
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from code.sohil.models import _netP, prob_data, generate_outlierexp_data, generate_classwise_data

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataroot', required=True, help='path to dataset')
##parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
#parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
#parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
#parser.add_argument('--ndf', type=int, default=64)
#parser.add_argument('--nc', type=int, default=1)
#parser.add_argument('--classindex', type=int, default=0)
#parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
#parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
#parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#parser.add_argument('--netP', default='netP_epoch_0.pth', help="path to netD (to continue training)")
#parser.add_argument('--outf', default='/fs/vulcan-scratch/krusinga/externalProjects/sohilGAN/ProbDistGAN/ryenExperiments/outputs/mnist_outlier_z_10_epoch_25',
#                    help='folder to output images and model checkpoints')
#parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
#parser.add_argument('--id', type=int, help='identifying number')
## added by Ryen:
#parser.add_argument('--fname',type=str,default='features.mat',help='name of training mat file with probs')
#parser.add_argument('--startProportions',type=str,default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')
#parser.add_argument('--endProportions',type=str,default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')
#parser.add_argument('--samples',type=int,default=10000)

class RegressorRun3Col(Operator):
  def __init__(self, config, args):
    super(RegressorRun3Col,self).__init__(config, args)
    opt = {
      'netP':'netP',
      'netPinstance':-1,
      'netPexpNum':-1,
      'startProportions':(0.1*np.ones(10)),
      'endProportions':(0.1*np.ones(10)),
      'samples':10000,
      'sigma':25
    }
    opt.update(args)
    self.opt = edict(opt)
    self.opt.startProportions = np.array(self.opt.startProportions)
    self.opt.endProportions = np.array(self.opt.endProportions)
    self.opt.startProportions[9] = 1-np.sum(self.opt.startProportions[:9])
    self.opt.endProportions[9] = 1-np.sum(self.opt.endProportions[:9])

    self.loader = self.dependencies[0]
    self.probs = np.array([])

  def run(self):
    self.log("Loading netP")
    netP = self.loader.getNetP(options=self.opt)
    netP.eval()

    self.log("Getting loaders")
    mnistPath = self.getPath('mnist')
    loaders = [ generate_classwise_data(mnistPath,0),
      generate_classwise_data(mnistPath,1),
      generate_classwise_data(mnistPath,2),
      generate_classwise_data(mnistPath,3),
      generate_classwise_data(mnistPath,4),
      generate_classwise_data(mnistPath,5),
      generate_classwise_data(mnistPath,6),
      generate_classwise_data(mnistPath,7),
      generate_classwise_data(mnistPath,8),
      generate_classwise_data(mnistPath,9)
    ]

    self.sampleShiftInterpolate(netP, loaders, self.opt.samples, self.opt.startProportions, self.opt.endProportions)


  def getAnalysisData(self):
    # Should convolve probabilities with gaussian or something
    xran = np.arange(-3*self.opt.sigma, 3*self.opt.sigma+1, 1)
    gaussFilter = norm.pdf(xran,0,self.opt.sigma)
    filteredProbs = np.convolve(self.probs, gaussFilter, 'valid') #Change to valid convolution
    plotDict = {
      'data':np.array([np.arange(len(filteredProbs)), filteredProbs]),
      'title':'Domain shift {0} --> {1}'.format(str(self.opt.startProportions),str(self.opt.endProportions)),
      'xlabel':'Input number',
      'ylabel':'Regressor Probability',
      'format':'png'
    }
    toPlot = Data(plotDict, 'lineplot', 'domainShiftCurve')

    return [toPlot]

  def getRandomFromDistr(self, loaders,probs):
    choice = nrand.choice(10,1,p=probs)[0]
    L = loaders[choice]
    choice2 = nrand.choice(len(L),1)
    (im, label) = L[choice2]
    return im, label #possibly need to unsqueeze

  def sampleShiftInterpolate(self, net, loaders, samples, startP, endP):
    self.log("Interpolating class distributions")
    images = np.empty((samples,3,28,28))
    labels = np.empty((samples))
    probs = np.empty((samples))

    changeVec = (endP-startP)/samples
    pvec = startP
    # Test line
    im, label = self.getRandomFromDistr(loaders,pvec)
    self.log(str(im.size()))
    # end test line
    for i in range(samples):
      if i%100 == 0:
        self.log("Sample %d"%i)
      pvec = pvec+changeVec
      im, label = self.getRandomFromDistr(loaders,pvec)
      im = torch.cat([im,im,im],1)
      images[i] = im.numpy()
      labels[i] = label
      im = Variable(im)
      probs[i] = net(im)[0]

    resultData= {
        'images':images.astype(np.float32),
        'label':labels.astype(int),
        'prob':probs.astype(np.float32)
      }
    self.log("Saving regressorResults")
    self.files.save(resultData,'regressorResults',saver='mat')

    self.probs = probs

