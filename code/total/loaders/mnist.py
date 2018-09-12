from LoaderTemplate import LoaderTemplate

import torch 
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from scipy.io import loadmat
import sys
from os.path import join

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


class LabeledMatLoader(data.Dataset):
  def __init__(self, matFile, distribution=None, outShape=None, returnLabel=False, mode='train'):
    self.matDirectory = matDirectory
    self.distribution = distribution
    self.outShape = outShape
    self.mode = mode
    self.returnLabel = returnLabel

    # Load train or test
    data = loadmat(matFile)
    if mode=='test':
      self.X = data['Xtest'].astype(np.float32)
      self.Y = data['Ytest'].astype(int).squeeze()
    else:
      self.X = data['Xtrain'].astype(np.float32)
      self.Y = data['Ytrain'].astype(int).squeeze()

    # Handle input / output shapes
    self.defaultShape = self.X[0].shape #
    if self.outShape != self.defaultShape:
      self.reshape = True
      assert(self.outShape[1] == self.defaultShape[1] and self.outShape[2] == self.defaultShape[2]) # Don't try to resize for the moment
#      self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Scale((self.outShape[1],self.outShape[2])), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    else:
      self.reshape = False

    # Get labels
    labels = np.unique(Y) #Should be sorted
    self.classwiseSubsets = [np.where(Y==lbl) for lbl in labels]

    # Resample X according to a distribution
    if distribution is not None:
      self.counts = [len(sub) for sub in classwiseSubsets]
      self.rangeInds = getProportionalIndices(distribution, counts, randomize=False)
      self.proportionalSubsets = [cw[rangeInds[i]] for i, cw in enumerate(classwiseSubsets)]
      self.totalIndices = np.random.permutation(np.concatenate(self.proportionalSubsets))
      self.X = self.X[self.totalIndices]
      self.Y = self.Y[self.totalIndices]

  def __len__(self):
    return len(self.X)

  def __getitem__(self,item):
    x = torch.from_numpy(self.X[item])
    if self.reshape:
      # Need to reshape x to have right number of colors, right scale
      if self.outShape[0] == 3 and x.size(0) == 1:
        x = torch.cat([x,x,x],dim=0)
      elif self.outShape[0]==1 and x.size(0) == 3:
        x = (x[0]*0.2989+x[1]*0.5870+x[2]*0.1140).unsqueeze(0)
#      x = self.transform(x)

    y = self.Y[item]
    if self.returnLabel:
      return x, y
    else:
      return x

  def sampleFromClassDistribution(self, distribution, number=1, squeeze=True):
    choices = np.random.choice(len(self.labels), size=number, p=distribution)
    inds = [np.random.choice(self.classwiseSubsets[i]) for i in choices]
    resultX = torch.Tensor(self.X[inds])
    resultY = torch.IntTensor(self.Y[inds])
    if number==1 and squeeze:
      resultX = torch.squeeze(resultX,dim=0)
      resultY = resultY[0]
    if self.returnLabel:
      return resultX, resultY
    else:
      return resultX

    

