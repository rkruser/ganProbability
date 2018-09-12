# Module to load the data and model

from mlworkflow import Loader
from easydict import EasyDict as edict
from copy import copy
import sys

# Torch and helper stuff
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
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

# May need to prefix the import as code.models?
from code.sohil.models import _netG, _netD, _netP, weights_init, generate_data, mog_netD, mog_netG, generate_outlierexp_data, generate_classwise_data, outlier2, prob_data_ryen, generate_mnist_distribution



class LoadGAN(Loader):
  def __init__(self, config, args):
    super(LoadGAN, self).__init__(config, args)
    opt = {
      # Loader default options
      'dataset':'mnist',
      'loadFromExperiment':-1,
      'batchSize':64,
      'workers':2,
      'imageSize':32,
      # Model default options
      'ngpu':1,
      'nz':20,
      'ngf':64,
      'ndf':64,
      'nc':1,
      'cuda':False,
      'netG':'',
      'netGexpNum':-1, #experiment number to load from
      'netGinstance':-1, #Epoch snapshot to load from
      'netD':'',
      'netDexpNum':-1, # experiment number to load from
      'netDinstance':-1, #epoch snapshot to load from
      'netP':'',
      'netPinstance':-1,
      'netPexpNum':-1,
      'lr':0.0002,
      'beta1':0.5,
      'manualSeed':None,
      'proportions':(0.1*np.ones(10))
    }
    opt.update(args)
    self.opt = edict(opt)
    self.opt.proportions = np.array(self.opt.proportions)
    self.opt.proportions[9] = 1-np.sum(self.opt.proportions[:9])

  # Does nothing on run
  # Because is called by later modules
  def run(self):
    pass

  def getProbData(self, options={}):
    opt = copy(self.opt)
    options = copy(options)
    opt.update(options)
    opt = edict(opt)

    # Need to modify prob_data in models
    datapath = self.getPath('samples',number=opt.loadFromExperiment, threadSpecific=False)
    trainset = prob_data_ryen(path=datapath, train=True)
    testset = prob_data_ryen(path=datapath, train=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    return trainloader, testloader

  # Return a torch dataloader object
  #  corresponding to the desired dataset
  def getLoader(self, options={}):
    opt = copy(self.opt)
    options = copy(options)
    opt.update(options)
    opt = edict(opt)

    dataset = opt.dataset
    dataroot = ''
    if dataset in ['imagenet', 'folder','lfw','lsun','cifar10','mnist','mnist_outlier','mnist_classwise', 'mnist_distribution']:
      try:
        # Path defined in master.yaml
        dataroot = self.getPath(dataset) 
      except KeyError:
        print "Dataroot for {0} not found".format(dataset)
        sys.exit()
   

    self.log('Getting dataset %s'%dataset)
    if dataset in ['imagenet', 'folder', 'lfw']:
      dataset = dset.ImageFolder(root=dataroot,
                                 transform=transforms.Compose([
                                     transforms.Scale(opt.imageSize),
                                     transforms.CenterCrop(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
    elif dataset == 'lsun':
      dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_train'],
                          transform=transforms.Compose([
                              transforms.Scale(opt.imageSize),
                              transforms.CenterCrop(opt.imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
    elif dataset == 'cifar10':
      dataset = dset.CIFAR10(root=dataroot, download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
    elif dataset == 'mnist':
      dataset = dset.MNIST(root=dataroot, download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
        # edited by Ryen to pad MNIST rather than scale it
    #    assert(opt.imageSize == 32)
    #    dataset = dset.MNIST(root=opt.dataroot, download=True,
    #                           transform=transforms.Compose([
    #                               transforms.Pad(2,fill=0),
    #                               transforms.ToTensor(),
    #                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                           ]))

    elif dataset == 'fake':
      dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                              transform=transforms.ToTensor())
    elif dataset == 'mog':
      dataset = generate_data(num_mode=8,except_num=2)
    elif dataset == 'mnist_outlier':
      dataset = generate_outlierexp_data(root=dataroot,
          # transforms not necessary for this one
                                         #transform=transforms.Compose([transforms.ToPILImage(),transforms.Scale(opt.imageSize),]),
                                         #transform2=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                                         size=opt.imageSize)
    elif dataset == 'outlier2':
      # Proportions for class distribution shift
      opt.proportions = np.array(opt.proportions)
      opt.proportions[9] = 1-np.sum(opt.proportions[:9]) # To ensure that they sum to 1 numerically
      dataset = outlier2(root=dataroot,size=opt.imageSize,proportions=opt.proportions)
    elif dataset == 'mnist_classwise':
      dataset = generate_classwise_data(root=dataroot, label=opt.classindex,
                                         transform=transforms.Compose([transforms.ToPILImage(),transforms.Scale(opt.imageSize),]),
                                         transform2=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                                         size=opt.imageSize)
    elif dataset == 'mnist_distribution':
      dataset = generate_mnist_distribution(datadir=dataroot, probs=opt.proportions)

    assert dataset

    self.log('Creating dataloader')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    return dataloader


  # Return the model corresponding to the option
  def getGANModel(self, options={}):
    opt = copy(self.opt)
    options = copy(options)
    opt.update(options)
    self.log("Using options:\n"+str(opt))
    opt = edict(opt)

    ngpu = opt.ngpu
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    nc = opt.nc

    if opt.manualSeed is None:
      self.log("Generating random seed")
      opt.manualSeed = random.randint(1, 10000)
    self.log("Using random seed %d"%opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True #??

    if torch.cuda.is_available() and not opt.cuda:
        self.log("WARNING: You have a CUDA device, so you should probably run with --cuda")

    self.log("Loading netG")
    if opt.dataset == 'mog':
        netG = mog_netG(ngpu, nz)
    else:
        netG = _netG(ngpu, nz,ngf,nc)
    netG.apply(weights_init)
    if opt.netG != '':
      self.log(" ...from file key %s"%opt.netG)
      netG.load_state_dict(self.files.load(opt.netG, instance=opt.netGinstance,
            number=opt.netGexpNum,loader='torch'))
    self.log("netG structure:")
    self.log(str(netG))

    self.log("Loading netD")
    if opt.dataset == 'mog':
        netD = mog_netD(ngpu)
    else:
        netD = _netD(ngpu,nc,ndf)
    netD.apply(weights_init)
    if opt.netD != '':
      self.log(" ...from file key %s"%opt.netD)
      netD.load_state_dict(self.files.load(opt.netD, instance=opt.netDinstance,
            number=opt.netDexpNum,loader='torch'))
    self.log("netD structure:")
    self.log(str(netD))

    criterion = nn.BCELoss()

    if opt.dataset == 'mog':
        input = torch.FloatTensor(opt.batchSize, 2)
        noise = torch.FloatTensor(opt.batchSize, nz)
#        fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_(0, 1)
    else:
        input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
        noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
#        fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
      self.log("Using cuda") 
      netD.cuda()
      netG.cuda()
      criterion.cuda()
      input, label = input.cuda(), label.cuda()
#        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
      noise = noise.cuda()

    # fixed_noise appears to be useless here
#    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    problem = {
      'netD':netD,
      'netG':netG,
      'optimD':optimizerD,
      'optimG':optimizerG,
      'criterion':criterion,
      'input':input,
      'label':label,
      'noise':noise,
      'real_label':real_label,
      'fake_label':fake_label,
      'opt':opt #Send the rest of the options in case trainer needs them
    }

    # Then need to return (net, criterion, optimizer, etc...)
    return problem

  def getSamplingGAN(self,options={}):
    opt = copy(self.opt)
    options = copy(options)
    opt.update(options)
    self.log("Using options:\n"+str(opt))
    opt = edict(opt)

    # Repeat code; move to __init__?
    if opt.manualSeed is None:
      self.log("Generating random seed")
      opt.manualSeed = random.randint(1, 10000)
    self.log("Using random seed %d"%opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True #??

    if torch.cuda.is_available() and not opt.cuda:
        self.log("WARNING: You have a CUDA device, so you should probably run with --cuda")

    netG = _netG(opt.ngpu, opt.nz, opt.ngf, opt.nc)
    self.log("Load netG from file key %s"%opt.netG)
    try:
      netG.load_state_dict(self.files.load(opt.netG, instance=opt.netGinstance,
            number=opt.netGexpNum,loader='torch'))
    except Exception as e:
      self.log("Problem loading netG from file")
      raise e
    self.log("netG structure:")
    self.log(str(netG))

    if opt.cuda:
      netG = netG.cuda()

    sampleProblem = {
      'netG':netG,
      'opt':opt
    }

    return sampleProblem

  def getRegressorProblem(self, options={}):
    opt = copy(self.opt)
    options = copy(options)
    opt.update(options)
    self.log("Using options:\n"+str(opt))
    opt = edict(opt)

    ngpu = opt.ngpu
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    nc = opt.nc

    if opt.manualSeed is None:
      self.log("Generating random seed")
      opt.manualSeed = random.randint(1, 10000)
    self.log("Using random seed %d"%opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True #??

    if torch.cuda.is_available() and not opt.cuda:
        self.log("WARNING: You have a CUDA device, so you should probably run with --cuda")

    self.log("Loading netP")
    if opt.dataset == 'mog':
        netP = mog_netP(ngpu)
    else:
        netP = _netP(ngpu,nc,ndf)
    netP.apply(weights_init)
    if opt.netP != '':
        netP.load_state_dict(self.files.load(opt.netP,instance=opt.netPinstance,
              number=opt.netPexpNum,loader='torch'))
    self.log("netP structure")
    self.log(str(netP))

    # criterion = nn.MSELoss(size_average=True)
    criterion = nn.SmoothL1Loss(size_average=True)

    if opt.cuda:
      netP.cuda()
      criterion.cuda()

    # setup optimizer
    optimizerP = optim.Adam(netP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0002)
    scheduler = lr_scheduler.StepLR(optimizerP, step_size=200, gamma=0.1)

    problem = {
      'netP':netP,
      'optimP':optimizerP,
      'scheduler':scheduler,
      'criterion':criterion,
      'opt':opt #Send the rest of the options in case trainer needs them
    }

    return problem

  def getNetP(self,options={}):
    opt = copy(self.opt)
    options = copy(options)
    opt.update(options)
    opt = edict(opt)

    if opt.dataset == 'mog':
      netP = mog_netP(opt.ngpu)
    else:
      netP = _netP(opt.ngpu,opt.nc,opt.ndf)

    netP.load_state_dict(self.files.load(opt.netP,instance=opt.netPinstance,
          number=opt.netPexpNum,loader='torch'))

    return netP


      

    
