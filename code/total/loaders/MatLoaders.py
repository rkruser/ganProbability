from copy import copy
from mlworkflow import Loader

import torch 
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from scipy.io import loadmat
import sys
from os.path import join

class LoaderTemplate(Loader):
  def __init__(self, config, args):
    super(LoaderTemplate, self).__init__(config,args)
    self.opt = {
      'batchSize':64,
      'workers':2,
      'shuffle':True
    }
    args = copy(args)
    self.opt.update(args)

    self.batchSize = self.opt['batchSize']
    self.workers = self.opt['workers']
    self.shuffle = self.opt['shuffle']
    self.path = None

  # Return the torch dataset object for the dataset
  # Outshape is the desired tensor shape
  # Everything will be normalized between -1 and 1
  # returnClass decides whether the dataset object will
  #  return a pair (x,y) or just x
  # This function should give an error if outShape is 
  #  not compatible with the dataset
  def getDataset(self, outShape = None, distribution=None, labels=None, mode='train', returnLabel = False):
    # Before returning, can check for compatible shapes
    return MatLoader(self.path, outShape = outShape, distribution=distribution, labels=labels, returnLabel=returnLabel, mode=mode)

  def getDataloader(self, outShape = None, distribution=None, labels=None, mode='train', returnLabel = False):
    # Before returning, can check for compatible shapes
    return data.DataLoader(MatLoader(self.path, outShape = outShape, distribution=distribution, labels=labels, returnLabel=returnLabel, mode=mode), batch_size = self.batchSize, shuffle=self.shuffle, num_workers = self.workers)


class MNISTSize28Cols1(LoaderTemplate):
  def __init__(self, config, args):
    super(MNISTSize28Cols1, self).__init__(config, args)
    self.path = self.getPath('mnist28') #Change depending on dataset

class MNISTSize32Cols1(LoaderTemplate):
  def __init__(self, config, args):
    super(MNISTSize32Cols1, self).__init__(config, args)
    self.path = self.getPath('mnist32') #Change depending on dataset

class MNISTSize64Cols1(LoaderTemplate):
  def __init__(self, config, args):
    super(MNISTSize64Cols1, self).__init__(config, args)
    self.path = self.getPath('mnist64') #Change depending on dataset

class CFAR10Size28Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CFAR10Size28Cols3, self).__init__(config, args)
    self.path = self.getPath('cifar10_28') #Change depending on dataset

class CFAR10Size32Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CFAR10Size32Cols3, self).__init__(config, args)
    self.path = self.getPath('cifar10_32') #Change depending on dataset

class CFAR10Size64Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CFAR10Size64Cols3, self).__init__(config, args)
    self.path = self.getPath('cifar10_64') #Change depending on dataset

class CUB200Size64Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CUB200Size64Cols3, self).__init__(config, args)
    self.path = self.getPath('cub200_64') #Change depending on dataset

class CUB200Size32Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CUB200Size32Cols3, self).__init__(config, args)
    self.path = self.getPath('cub200_32') #Change depending on dataset

class BirdsnapSize64Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CUB200Size64Cols3, self).__init__(config, args)
    self.path = self.getPath('birdsnap64')

class BirdsnapSize32Cols3(LoaderTemplate):
  def __init__(self, config, args):
    super(CUB200Size32Cols3, self).__init__(config, args)
    self.path = self.getPath('birdsnap32')


class ProbData(LoaderTemplate):
  def __init__(self, config, args):
    super(ProbData, self).__init__(config, args)
    args = copy(args)
    self.opt = {
      'sampleKey':'samples',
      'sampleExpNum':-1
    }
    self.opt.update(args)
    for a in self.opt:
      setattr(self, a, self.opt[a])

    self.current = None #??
    self.path = self.getPath(self.sampleKey, number=self.sampleExpNum, threadSpecific=False) #Need to add to config

  def getDataset(self, deep=False, mode='train', trainProportion=0.8, outShape=None):
    return ProbLoader(self.path, deep=deep, mode=mode, trainProportion=trainProportion)

  def getDataloader(self, deep=False, mode='train', trainProportion=0.8, outShape=None):
    return data.DataLoader(ProbLoader(self.path, deep=deep, mode=mode, trainProportion=trainProportion), batch_size = self.batchSize, shuffle=self.shuffle, num_workers = self.workers)


# Assumes you have a matfile
# containing Xtrain, (Ytrain), Xtest, (Ytest) of the appropriate sizes
class MatLoader(data.Dataset):
  def __init__(self, matFile, outShape=None, distribution=None, labels=None,  returnLabel=False, mode='train'):
    self.matFile = matFile
    self.distribution = distribution
    self.outShape = outShape
    self.mode = mode
    self.returnLabel = returnLabel

    self.X = None
    self.Y = None

    # Load train or test
    data = loadmat(matFile)
    if mode=='test':
      self.X = data['Xtest'].astype(np.float32)
      if 'Ytest' in data:
        self.Y = data['Ytest'].squeeze()
    else:
      self.X = data['Xtrain'].astype(np.float32)
      if 'Ytrain' in data:
        self.Y = data['Ytrain'].squeeze()

    if returnLabel:
      assert(self.Y is not None)

    # Handle input / output shapes
    self.defaultShape = self.X[0].shape #
    if (self.outShape is not None) and (self.outShape != self.defaultShape):
      self.reshape = True
      assert(self.outShape[1] == self.defaultShape[1] and self.outShape[2] == self.defaultShape[2]) # Don't try to resize for the moment
#      self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Scale((self.outShape[1],self.outShape[2])), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
      self.reshape = False

    # Get labels
    if self.Y is not None:
      if labels is not None:
        self.labels = labels
      else:
        self.labels = np.unique(self.Y) #Is sorted
      self.classwiseSubsets = [np.where(self.Y==lbl) for lbl in self.labels]
    else:
      self.labels = None
      self.classwiseSubsets = None

    # Resample X according to a distribution
    if distribution is not None:
      assert(self.Y is not None)
      self.counts = np.array([len(sub) for sub in classwiseSubsets])
      assert(np.all(self.counts>0)) #Must be true to have distribution
      self.rangeInds = getProportionalIndices(distribution, counts, randomize=False)
      self.proportionalSubsets = [cw[rangeInds[i]] for i, cw in enumerate(classwiseSubsets)]
      self.totalIndices = np.random.permutation(np.concatenate(self.proportionalSubsets))
      self.X = self.X[self.totalIndices]
      self.Y = self.Y[self.totalIndices]

  def __len__(self):
    return len(self.X)

  def __getitem__(self,item):
    x = torch.from_numpy(self.X[item]) # Do I want from_numpy? Dataset could be modified
    if self.reshape:
      # Need to reshape x to have right number of colors, right scale
      if self.outShape[0] == 3 and x.size(0) == 1:
        x = torch.cat([x,x,x],dim=0)
      elif self.outShape[0]==1 and x.size(0) == 3:
        x = (x[0]*0.2989+x[1]*0.5870+x[2]*0.1140).unsqueeze(0)
#      x = self.transform(x)

    if self.returnLabel:
      y = self.Y[item]
      return x, y
    else:
      return x

  def sampleFromClassDistribution(self, distribution, number=1, squeeze=True):
    assert(self.Y is not None)
    choices = np.random.choice(len(self.labels), size=number, p=distribution)
    inds = np.array([np.random.choice(self.classwiseSubsets[i]) for i in choices])
    resultX = torch.Tensor(self.X[inds])
    resultY = torch.IntTensor(self.Y[inds])
    if number==1 and squeeze:
      resultX = torch.squeeze(resultX,dim=0)
      resultY = resultY[0]
    if self.returnLabel:
      return resultX, resultY
    else:
      return resultX

    
# Utility function for getting indices according to proportions
def getProportionalIndices(distribution, counts,randomize=False):
  assert(len(distribution) == len(counts))
  total = np.sum(counts)
  sizes = np.array(float(total)*distribution, dtype='int')
  if randomize:
    ranges = [np.random.choice(counts[i],size=sizes[i],replace=True) for i in range(len(sizes))]
  else:
    ranges = [np.arange(s)%counts[i] for i,s in enumerate(sizes)]
  return ranges

class ProbLoader(data.Dataset):
  def __init__(self, matpath, deep=False, mode='train', trainProportion=0.8):
    self.data = loadmat(matpath)
    if deep:
      self.X = self.data['feats'] #Actually no, but okay
    else:
      self.X = self.data['images']
    self.Y = self.data['prob'].squeeze()

    if mode=='train':
      self.train = True
    else:
      self.train = False

    self.outShape = self.X[0].shape 

    self.dataSize = len(self.X)
    self.trainSize = int(self.dataSize*trainProportion)
    self.testSize = self.dataSize-self.trainSize
    
    self.Xtrain = self.X[:self.trainSize]
    self.Ytrain = self.Y[:self.trainSize]
    self.Xtest = self.X[self.trainSize:]
    self.Ytest = self.Y[self.trainSize:]


  def __len__(self):
    if self.train: 
      return self.trainSize
    else:
      return self.testSize

  def __getitem__(self, item):
    if self.train:
      return torch.from_numpy(self.Xtrain[item]), self.Ytrain[item]
    else:
      return torch.from_numpy(self.Xtest[item]), self.Ytest[item]

  def setMode(self, mode):
    if mode == 'train':
      self.train = True
    else:
      self.train = False

  def getOutshape(self):
    return self.outShape
