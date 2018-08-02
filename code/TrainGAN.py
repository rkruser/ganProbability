##### mlworkflow
from mlworkflow import Operator, Data
from easydict import EasyDict as edict

#from __future__ import print_function
#import argparse
import json
import numpy as np
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from code.models import _netG, _netD, weights_init, generate_data, mog_netD, mog_netG, generate_outlierexp_data, generate_classwise_data, outlier2


#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='cifar10 | mnist | lsun | imagenet | folder | lfw | fake')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
#parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
#parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
#parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
#parser.add_argument('--ngf', type=int, default=64)
#parser.add_argument('--ndf', type=int, default=64)
#parser.add_argument('--nc', type=int, default=3)
#parser.add_argument('--classindex', type=int, default=0)
#parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
#parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
#parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
#parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
#parser.add_argument('--netG', default='', help="path to netG (to continue training)")
#parser.add_argument('--netD', default='', help="path to netD (to continue training)")
#parser.add_argument('--outf', default='/fs/vulcan-scratch/krusinga/distGAN/checkpoints',
#                    help='folder to output images and model checkpoints')
#parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument('--proportions',type=str, help='Probabilities of each class in mnist',default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')
#opt = parser.parse_args()
#print(opt)

class TrainGAN(Operator):
  def __init__(self, config, args):
    super(TrainGAN,self).__init__(config,args)
    # Default training options
    opt = {
      'nepochs':1
    }
    opt.update(args)
    self.opt = edict(opt)
    assert(len(self.dependencies) > 0)
    self.loader = self.dependencies[0] #Depend on loader class

    self.errG = []
    self.errD = []

  def run(self):
    # Place the following in loader
#    opt = self.opt

    self.log("Getting dataloader")
    dataloader = self.loader.getLoader()
    self.log("Getting model")
    gan = self.loader.getGANModel()
    # Need to get data and models from loader here
    
    netD = gan['netD']
    netG = gan['netG']
    input = gan['input']
    label = gan['label']
    noise = gan['noise']
    real_label = gan['real_label']
    fake_label = gan['fake_label']
    optimizerG = gan['optimG']
    optimizerD = gan['optimD']
    opt = gan['opt']
    criterion = gan['criterion']
    opt.update(self.opt)
    opt = edict(opt)

    nz = opt.nz

    self.log("Beginning training")
    for epoch in range(opt.nepochs):
        self.log("===Begin epoch %d"%epoch)
        for i, data in enumerate(dataloader, 0): #what's with the 0?
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real

            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)

            output = netD(inputv)

            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            if opt.dataset == 'mog':
                noise.resize_(batch_size, nz).normal_(0, 1)
            else:
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = netD(fake.detach()) # this just saves computation time
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = netD(fake) #Notable: no detach here
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            # Log and/or print the following
            self.log('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.nepochs, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            self.errG.append(errG.data[0])
            self.errD.append(errD.data[0])

                                                                      
    # do checkpointing
    # No instance indices here apparently (no epoch-wise saving)
    self.log("Saving netG")
    self.files.save(netG.state_dict(), 'netG', saver='torch')
    self.log("Saving netD")
    self.files.save(netD.state_dict(), 'netD', saver='torch')

    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, opt.classindex))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, opt.classindex))

  def getAnalysisData(self):
    assert(len(self.errG) == len(self.errD))
    errInfo = {
      'data':np.array([range(len(self.errG)),self.errG,self.errD]),
      'legend':['Generator error','Discriminator Error'],
      'xlabel':'Iteration (sub-epoch)',
      'ylabel':'Error',
      'title':'GAN training curve',
      'format':'png'
    }
    errAnalysis = Data(errInfo, 'lineplot', 'ganTrainPlot')
    return [errAnalysis]



# Todo:
   # Move rest of preamble to loader
   # Call getLoader and getModel here
   # Put necessary options in config file
   # Add data logging and saving for analysis
   # Test and tune
   # Then, move on and do the sampler and regressor
   # By the end of the day, be able to run a complete pipeline
