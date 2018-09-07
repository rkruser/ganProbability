from code.sohil.models import _netP, generate_classwise_data, prob_data, generate_outlierexp_data

from mlworkflow import Operator, Data
from easydict import EasyDict as edict

import numpy as np
import json
import scipy.io as sio
from scipy.stats import norm
import numpy.random as nrand
#import os.path as osp
#import argparse
import torch
import torch.nn as nn
#from torch.utils.data import DataLoader
from torch.autograd import Variable

def randomSelect(loader, n):
  choices = np.random.choice(len(loader), n, replace=True)
  objs = []
  for i in range(n):
    objs.append(loader[choices[i]][0].unsqueeze(0))
  return torch.cat(objs)

def meanVariance(net, loader, batchSize, iters):
  means = []
  for i in range(iters):
    print i
    x = Variable(randomSelect(loader, batchSize))
    p = net(x)
    p = p.data.numpy().squeeze()
    means.append(np.mean(p))
  meanMean = np.mean(means)
  meanStd = np.std(means)

  return meanMean, meanStd
    
    
    

  










