# Module to load the data and model
from mlworkflow import Loader
from easydict import EasyDict as edict
from copy import copy
import sys

# Torch and helper stuff
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz

from code.stackgan.datasets import TextDataset

class StackGANLoad(Loader):
  def __init__(self, config, args):
    super(StackGANLoad, self).__init__(config, args)
    opt = {
      # Loader default options
      'dataset':'cub',
      'embeddingType': 'cnn-rnn',
      'stage': 1,
      'batchSize':64,
      'workers':4,
      'imageSize':64,
      # Model default options
      'ngpu':1,
      'cuda':False, #?
      'zDim': 100,
      'netG':'',
      'netGexpNum':-1, #experiment number to load from
      'netGinstance':-1, #Epoch snapshot to load from
      'netD':'',
      'netDexpNum':-1, # experiment number to load from
      'netDinstance':-1, #epoch snapshot to load from
      'snapshotInterval': 20,
      'lr':0.0002,
      'maxEpoch': 600,
      'lrDecayEpoch': 20,
      'klCoeff': 2.0,
      'ganConditionDim': 128,
      'ganDfDim': 96,
      'ganGfDim': 192,
      'textDim': 1024,
      'beta1':0.5,
      'manualSeed':None,
    }
    opt.update(args)
    self.opt = edict(opt)

  # Does nothing on run
  # Because is called by later modules
  def run(self):
    pass

  def getLoader(self):
    seed = self.opt.manualSeed
    if seed is None:
      seed = random.randint(1,10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if self.opt.cuda:
      torch.cuda.manual_seed_all(seed)

    image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            # ToTensor permutes dimensions so c is first
            # Also transforms into range (0,1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # The normalize command pushes everything
            # into (-1,1)
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          imsize=cfg.IMSIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=self.opt.batchSize * self.opt.ngpu,
        drop_last=True, shuffle=True, num_workers=int(self.opt.workers))

    return dataloader

  # Return a dict containing generator, discriminator,
    # optimizer, and other training entities
  def getProblem(self):
    pass

