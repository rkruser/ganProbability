from mlworkflow import Operator, Data
from easydict import EasyDict as edict

import numpy as np

#from __future__ import print_function
#import argparse
#import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from code.models import _netP, prob_data, weights_init, mog_netP

# Need to change prob_data to load samples.mat

# used for logging to TensorBoard
#from tensorboard_logger import configure, log_value


#parser = argparse.ArgumentParser()
#parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
#parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
#parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
#parser.add_argument('--ndf', type=int, default=64)
#parser.add_argument('--nc', type=int, default=3)
#parser.add_argument('--classindex', type=int, default=0)
#parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
#parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
#parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#parser.add_argument('--netP', default='', help="path to netD (to continue training)")
#parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
#                    help='folder to output images and model checkpoints')
#parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
#parser.add_argument('--id', type=int, help='identifying number')
## added by Ryen:
#parser.add_argument('--fname',type=str,default='features.mat',help='name of training mat file with probs')


class RegressorTraining(Operator):
  def __init__(self, config, args):
    super(RegressorTraining,self).__init__(config,args)
    opt = {}
    opt.update(args)
    self.opt = edict(opt)
    self.loader = self.dependencies[0]
    self.lossCurve = [[],[],[]]
    self.errorCurve = [[],[],[]]

  def run(self):
    trainloader, testloader = self.loader.getProbData()
    problem = self.loader.getRegressorProblem()
    netP = problem['netP']
    optimizerP = problem['optimP']
    criterion = problem['criterion']
    scheduler = problem['scheduler']
    opt = problem['opt']
   
    for epoch in range(self.opt.nepochs):
      self.log("Epoch: %d"%epoch)
      scheduler.step()
      self.train(trainloader, netP, optimizerP, criterion, opt.cuda, epoch)
      self.test(testloader, netP, criterion, opt.cuda, epoch)

      self.lossCurve[0].append(epoch)
      self.errorCurve[0].append(epoch)

    # do checkpointing
    self.log("Saving netP")
    self.files.save(netP.state_dict(), 'netP', saver='torch')

  def train(self, dataloader, net, optimizer, criterion, use_cuda, epoch):
      losses = AverageMeter()
      abserror = AverageMeter()

      net.train()
      for i, (data, label) in enumerate(dataloader):
          if use_cuda:
              data = data.cuda()
              label = label.cuda()
          optimizer.zero_grad()
          datav = Variable(data)
          labelv = Variable(label)

          output = net(datav)#, 5) #why the 5?
          err = criterion(output, labelv)

          losses.update(err.data[0], data.size(0))
          abserror.update((output.data - label).abs_().mean(), data.size(0))

          err.backward()
          optimizer.step()

      self.log('Training epoch %d, loss = %d, abserror=%d'%(epoch,losses.avg,abserror.avg))
#      self.trainCurve[0].append(epoch)
      self.lossCurve[1].append(losses.avg)
      self.errorCurve[1].append(abserror.avg)

  def test(self, dataloader, net, criterion, use_cuda, epoch):
      losses = AverageMeter()
      abserror = AverageMeter()
      net.eval()

      for i, (data, label) in enumerate(dataloader):
          if use_cuda:
              data = data.cuda()
              label = label.cuda()
          datav = Variable(data, volatile=True)
          labelv = Variable(label, volatile=True)

          output = net(datav)#, 5)
          err = criterion(output, labelv)

          losses.update(err.data[0], data.size(0))
          abserror.update((output.data - label).abs_().mean(), data.size(0))

      self.log('Testing epoch %d, loss = %d, abserror=%d'%(epoch,losses.avg,abserror.avg))
#      self.testCurve[0].append(epoch)
      self.lossCurve[2].append(losses.avg)
      self.errorCurve[2].append(abserror.avg)

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

