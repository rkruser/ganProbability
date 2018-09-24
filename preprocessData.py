import os.path as osp
import sys
import argparse
import numpy as np

import PIL
import torchvision.datasets as dset
import torchvision.transforms as transforms
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--outSize', type=int, default=28)


def getDataset(name, folder, size):
  dataset = None
  testset = None
  if name == 'lsun':
    dataset = dset.LSUN(db_path=folder, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(size),
                            transforms.CenterCrop(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
  elif name == 'cifar10':
    dataset = dset.CIFAR10(root=folder, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    testset = dset.CIFAR10(root=folder, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

  elif name == 'mnist':
    dataset = dset.MNIST(root=folder, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    testset = dset.MNIST(root=folder, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

  else:
    # Need to make sure extensions work
    dataset = dset.ImageFolder(root=folder,
#                               extensions=['png','jpg'],
                               transform=transforms.Compose([
                                   transforms.Resize((size,size), interpolation = PIL.Image.LANCZOS), #why scale before centercrop?
                            #       transforms.CenterCrop(size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

  if testset is not None:
    return (dataset, testset)
  else:
    return dataset


def getNumpy(dset, start, end):
  a = []
  b = []

  Yvals = False
  if isinstance(dset[0], tuple):
    Yvals = True
    
  for i in range(start, end):
    if i%10 == 0:
      print "Iter {}".format(i)
    try: #Need to test try/catch
      item = dset[i]
      if Yvals:
        a.append(np.array(item[0]))
        if item[1] is not None:
          b.append(item[1])
      else:
        a.append(np.array(item))
    except IOError as e:
      print "Error,",e 
      print "Skipping {0}".format(i)


  if Yvals and len(b) == len(a):
    return np.array(a), np.array(b)
  else:
    return np.array(a)


def main():
  opt = parser.parse_args()

  print "Getting dataset"
  dataset = getDataset(opt.name, opt.folder, opt.outSize)

  Xtrain = None
  Xtest = None
  Ytrain = None
  Ytest = None

  if isinstance(dataset, tuple):
    trainset = dataset[0]
    testset = dataset[1]
    Xtrain = getNumpy(trainset, 0, len(trainset))
    Xtest = getNumpy(testset, 0, len(testset))
    if isinstance(Xtrain, tuple):
      Xtrain, Ytrain = Xtrain
    if isinstance(Xtest, tuple):
      Xtest, Ytest = Xtest

  else:
    # LATER NEED TO RANDOMIZE TRAIN AND TEST
    nTrain = int(0.8*len(dataset))
    X = getNumpy(dataset, 0, len(dataset))
    Y = None
    if isinstance(X, tuple):
      X, Y = X
    choices = np.random.choice(len(X), size=nTrain, replace=False)
    inds = np.zeros(len(X),dtype=bool)
    inds[choices] = True
    Xtrain = X[inds]
    Xtest = X[np.logical_not(inds)]
    if Y is not None:
      Ytrain = Y[inds]
      Ytest = Y[np.logical_not(inds)]
  
  processed = {}
  if Xtrain is not None:
    processed['Xtrain'] = Xtrain
  if Xtest is not None:
    processed['Xtest'] = Xtest
  if Ytrain is not None:
    processed['Ytrain'] = Ytrain
  if Ytest is not None:
    processed['Ytest'] = Ytest
   
  print "Saving"
  savemat(osp.join(opt.folder, opt.name+str(opt.outSize)+'.mat'), processed)

if __name__=='__main__':
  main()
