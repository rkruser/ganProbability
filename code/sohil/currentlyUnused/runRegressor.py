import numpy as np
import json
import scipy.io as sio
import numpy.random as nrand
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import _netP, prob_data, generate_outlierexp_data, generate_classwise_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--classindex', type=int, default=0)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netP', default='netP_epoch_0.pth', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/krusinga/externalProjects/sohilGAN/ProbDistGAN/ryenExperiments/outputs/mnist_outlier_z_10_epoch_25',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number')
# added by Ryen:
parser.add_argument('--fname',type=str,default='features.mat',help='name of training mat file with probs')
parser.add_argument('--startProportions',type=str,default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')
parser.add_argument('--endProportions',type=str,default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')
parser.add_argument('--samples',type=int,default=10000)


opt = parser.parse_args()
netFileName = osp.join(opt.outf,opt.netP)
netP = _netP(opt.ngpu,opt.nc,opt.ndf)
netP.load_state_dict(torch.load(netFileName))
netP.eval()

opt.startProportions = np.array(json.loads(opt.startProportions))
opt.startProportions[9] = 1-np.sum(opt.startProportions[:9])
opt.endProportions = np.array(json.loads(opt.endProportions))
opt.endProportions[9] = 1-np.sum(opt.endProportions[:9])


loaders = [ generate_classwise_data(opt.dataroot,0),
  generate_classwise_data(opt.dataroot,1),
  generate_classwise_data(opt.dataroot,2),
  generate_classwise_data(opt.dataroot,3),
  generate_classwise_data(opt.dataroot,4),
  generate_classwise_data(opt.dataroot,5),
  generate_classwise_data(opt.dataroot,6),
  generate_classwise_data(opt.dataroot,7),
  generate_classwise_data(opt.dataroot,8),
  generate_classwise_data(opt.dataroot,9)
]

def getRandom(loaders,prob0):
  probRest = (1.-prob0)/9.
  probs = [prob0, probRest, probRest, probRest, probRest,
       probRest, probRest, probRest, probRest, probRest]
  choice = nrand.choice(10,1,p=probs)[0]
  L = loaders[choice]
  choice2 = nrand.choice(len(L),1)
  (im, label) = L[choice2]
  return im, label #possibly need to unsqueeze

def getRandomFromDistr(loaders,probs):
  choice = nrand.choice(10,1,p=probs)[0]
  L = loaders[choice]
  choice2 = nrand.choice(len(L),1)
  (im, label) = L[choice2]
  return im, label #possibly need to unsqueeze

def sampleShiftInterpolate(net, samples, startP, endP):
  images = np.empty((samples,1,28,28))
  labels = np.empty((samples))
  probs = np.empty((samples))

  changeVec = (endP-startP)/samples
  pvec = startP
  for i in range(samples):
    print "Sample", i
    pvec = pvec+changeVec
    im, label = getRandomFromDistr(loaders,pvec)
    images[i] = im.numpy()
    labels[i] = label
    im = Variable(im)
    probs[i] = net(im)[0]
  
  sio.savemat('ProbShift.mat',
      {
        'images':images.astype(np.float32),
        'label':labels.astype(int),
        'prob':probs.astype(np.float32)
        }
      )


def sampleShift(net, samples, rate):
  prob0 = 0
  images = np.empty((samples,1,28,28))
  labels = np.empty((samples))
  probs = np.empty((samples))
  for i in range(samples):
    print "Sample", i
    prob0 = prob0+rate
    im, label = getRandom(loaders,prob0)
    images[i] = im.numpy()
    labels[i] = label
    im = Variable(im)
    probs[i] = net(im)[0]
  
  sio.savemat('ProbShift.mat',
      {
        'images':images.astype(np.float32),
        'label':labels.astype(int),
        'prob':probs.astype(np.float32)
        }
      )

#sampleShift(netP, 10000, 0.0001)
    
print opt.startProportions, opt.endProportions
sampleShiftInterpolate(netP, opt.samples, opt.startProportions, opt.endProportions)

