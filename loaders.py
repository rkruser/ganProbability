# loaders
import torch
from torch.utils import data
from torch.utils.data import DataLoader #random_split, DataLoader
from scipy.io import loadmat
import numpy as np

locations={
	'mnist':'/vulcan/scratch/krusinga/mnist/mnist32.mat',
	'cifar10':'/vulcan/scratch/krusinga/cifar10/cifar10_32.mat',
#	'lsun':...,
	'birdsnap':'/vulcan/scratch/krusinga/birdsnap/birdsnap/download/images/birdsnap32.mat',
	'cub':'/vulcan/scratch/krusinga/CUB_200_2011/images/cub_200_2011_32.mat'
 # (omniglot) japanese_hiragana_32: /vulcan/scratch/krusinga/omniglot/omniglot/python/images_background/Japanese_(hiragana)/japanese_hiragana32.mat	
}

class TrainSplit(data.Dataset):
  def __init__(self, underlyingDset, indices):
    self.indices = indices
    self.dset = underlyingDset

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, i):
    return self.dset[self.indices[i]]

def random_split(dset, prop):
  n = len(dset)
  l1 = int(prop*n)
  permute = np.random.permutation(n)
  return TrainSplit(dset, permute[:l1]), TrainSplit(dset, permute[l1:])



# Assumes you have a matfile
# containing Xtrain, (Ytrain), Xtest, (Ytest) of the appropriate sizes
class MatLoader(data.Dataset):
  def __init__(self, matFile, outShape=None, distribution=None, labels=None, returnLabel=False, mode='train', fuzzy=False):
    self.matFile = matFile
    self.distribution = distribution
    self.outShape = outShape
    self.mode = mode
    self.returnLabel = returnLabel
    self.fuzzy = fuzzy # Add small random noise to pixel values and shrink away from boundaries

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
     # self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Scale((self.outShape[1],self.outShape[2])), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
      self.reshape = False

    # Get labels
    if self.Y is not None:
      if labels is not None:
        self.labels = labels
      else:
        self.labels = np.unique(self.Y) #Is sorted
      self.classwiseSubsets = [np.where(self.Y==lbl)[0] for lbl in self.labels]
    else:
      self.labels = None
      self.classwiseSubsets = None

    # Resample X according to a distribution
    if distribution is not None:
      assert(self.Y is not None)
      self.counts = np.array([len(sub) for sub in self.classwiseSubsets])
      assert(np.all(self.counts>0)) #Must be true to have distribution
      self.rangeInds = getProportionalIndices(distribution, self.counts, randomize=False)
      self.proportionalSubsets = [cw[self.rangeInds[i]] for i, cw in enumerate(self.classwiseSubsets)]
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
     # x = self.transform(x)

    # Fix this
    if self.fuzzy:
      x = x.add_(torch.Tensor(x.size()).uniform_(-1.0/255,1.0/255)).clamp_(-1,1)

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
    self.Y = self.data['probs'].squeeze()

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


# ************* MOG stuff ***************

def getMogMeans(nmeans=8, scale=1.0, omit=None):
  mogMeans = scale*np.array([[np.cos(i*2*np.pi/nmeans), np.sin(i*2*np.pi/nmeans)] for i in range(nmeans) if i != omit]).astype(np.float32)
  return mogMeans

def getRandomMeans(mogMeans, npts):
  indices = np.random.choice(len(mogMeans), npts, replace=True)
  return torch.from_numpy(mogMeans[indices])

def mogData(n, mogMeans=np.array([1.0]), stdev=0.05):
  # Get n samples of a mog circle
  data = torch.Tensor(n,2).normal_(0,stdev)+getRandomMeans(mogMeans,n)
  return data

class DataGenerator(object):
  def __init__(self, numBatches, batchsize, dataFunc):
    self.numBatches = numBatches
    self.batchsize = batchsize
    self.dataFunc = dataFunc
    self.index = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.index < self.numBatches:
      self.index += 1
      return self.dataFunc(self.batchsize)
    if self.index == self.numBatches:
      self.index = 0
    raise StopIteration()

  next = __next__

  def __len__(self):
    return self.numBatches



def getLoaders(loader='mnist', nc=3, size=32, root=None, batchsize=64, returnLabel=False,
	distribution=None, fuzzy=False, mode='train', validation=False, trProp=0.8, deep=False, stdev=0.05,
  shuffle=True):
  if root is None and loader not in ['mogSeven', 'mogEight']:
  	root = locations[loader]

  outshape = (nc, size, size)

  if loader == 'probdata':
  	dset =  ProbLoader(root, deep=deep, mode=mode)
  elif loader == 'mogEight':
    eightMeans = getMogMeans()
    return DataGenerator(1000, batchsize, lambda n : mogData(n, mogMeans=eightMeans, stdev=stdev))
  elif loader == 'mogSeven':
    eightMinusOne = getMogMeans(omit=2)
    return DataGenerator(1000, batchsize, lambda n : mogData(n, mogMeans=eightMinusOne, stdev=stdev))
  else:
    dset = MatLoader(root, outShape=outshape, distribution=distribution, returnLabel=returnLabel, mode=mode, fuzzy=fuzzy)

  if validation:
  #		trLen = int(float(trProp)*len(dset))
  #		valLen = len(dset)-trLen
    trainDset, valDset = random_split(dset, trProp)
    return (DataLoader(trainDset, batch_size=batchsize, shuffle=shuffle), DataLoader(valDset, batch_size=batchsize, shuffle=False))
  else:
    return DataLoader(dset, batch_size=batchsize, shuffle=shuffle)










