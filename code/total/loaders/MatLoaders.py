from code.total.loaders.LoaderTemplate import LoaderTemplate

import torch 
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from scipy.io import loadmat
import sys
from os.path import join


class MNISTSize28Cols1(LoaderTemplate):
  def __init__(self, config, args):
    super(LoaderTemplate, self).__init__(config, args)
    self.path = self.getPath('mnist')

  def getDataset(self, outShape = None, distribution=None, labels=None, mode='train', returnClass = False):
    return MatLoader(self.path, outShape = outShape, distribution=distribution, labels=labels, returnLabel=returnLabel, mode=mode)



# Assumes you have a matfile
# containing Xtrain, (Ytrain), Xtest, (Ytest) of the appropriate sizes
class MatLoader(data.Dataset):
  def __init__(self, matFile, outShape=None, distribution=None, labels=None,  returnLabel=False, mode='train'):
    self.matDirectory = matDirectory
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
        self.Y = data['Ytest'].astype(int).squeeze()
    else:
      self.X = data['Xtrain'].astype(np.float32)
      if 'Ytrain' in data:
        self.Y = data['Ytrain'].astype(int).squeeze()

    if returnLabel:
      assert(self.Y is not None)

    # Handle input / output shapes
    self.defaultShape = self.X[0].shape #
    if self.outShape != self.defaultShape:
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
        self.labels = np.unique(Y) #Should be sorted
      self.classwiseSubsets = [np.where(Y==lbl) for lbl in self.labels]
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

