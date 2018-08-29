from mlworkflow import Loader
from easydict import EasyDict as edict

import torch.utils.data as data
import torchvision.transforms as transforms
#from os.path import join
from scipy.io import loadmat
import numpy as np
import torch
import sys
from code.sohil.models import generate_mnist_distribution
from code.stackgan.datasets import TextDataset

matfile = '/vulcan/scratch/krusinga/birdsnap/birdsnap/download/birdsnap.mat'

class BirdsnapDataset(data.Dataset):
  def __init__(self, matfile):
    self.matfile = matfile
    data = loadmat(self.matfile)
    self.fnames = data['names']
    self.images = data['images']

  def __len__(self):
    return len(self.images)

  def __getitem__(self, item):
    im = self.images[item]
    im = np.array(im, dtype='float32')
    im = im/255.0
    im = (im-0.5)/0.5
    im = torch.Tensor(im) #maybe more efficient if done using torch in-place
    return im


# Sampler for classwise training of GAN
# Technically every epoch will have slightly different data due
#  to off-sync wrap-around and such, but does that matter
# Okay, this is a terrible setup
#class generate_mnist_distribution(data.Dataset):
#  def __init__(self, datadir, probs=(0.1*np.ones(10)), length=60000):
#    self.datadir = datadir
#    self.probs = probs
#    self.length = length
#    self.loaders = [generate_classwise_data(self.datadir,i) for i in range(10)]
#    self.counters = np.zeros(10,dtype=int)
#    self.choices = np.random.choice(10,self.length,p=self.probs)
#
#  def __len__(self):
#    return self.length
#
#  def __getitem__(self, item):
#    choice = self.choices[item]
#    data,_ = self.loaders[choice][self.counters[choice]]
#    self.counters[choice] = (self.counters[choice] + 1)%len(self.loaders[choice])
#
#    return data, choice

class DataloaderRyen(Loader):
  def __init__(self, config, args):
    super(DataloaderRyen, self).__init__(config, args)
    opt = {
      'dataset':'birdsnap',
      'matfile':'/vulcan/scratch/krusinga/birdsnap/birdsnap/download/birdsnap.mat',
      'imsize':64,
      'batchSize':64,
      'workers':2,
      'proportions':(0.1*np.ones(10))
    }
    opt.update(args)
    self.opt = edict(opt)

  def getLoader(self):
    dset = self.opt.dataset
    if dset == 'birdsnap':
      return self.getBirdsnap()
    elif dset == 'mnist':
      return self.getMnist()
    elif dset == 'cub':
      return self.getCub()

  def getDataset(self):
    dset = self.opt.dataset
    if dset == 'birdsnap':
      dataset = BirdsnapDataset(self.opt.matfile),
    elif dset == 'mnist':
      dataset = generate_mnist_distribution(datadir=self.getPath('mnist'), probs=self.opt.proportions)
    elif dset == 'cub':
      image_transform = transforms.Compose([
        transforms.RandomCrop(self.opt.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        # ToTensor permutes dimensions so c is first
        # Also transforms into range (0,1)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # The normalize command pushes everything
        # into (-1,1)
      dataset = TextDataset(self.getPath('cub'), 'train',
                      imsize=self.opt.imsize,
                      transform=image_transform)
    return dataset


  def getCub(self):
    image_transform = transforms.Compose([
            transforms.RandomCrop(self.opt.imsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            # ToTensor permutes dimensions so c is first
            # Also transforms into range (0,1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # The normalize command pushes everything
            # into (-1,1)
    dataset = TextDataset(self.getPath('cub'), 'train',
                          imsize=self.opt.imsize,
                          transform=image_transform)

    dataloader = data.DataLoader(
        dataset, batch_size=self.opt.batchSize,
        drop_last=False, shuffle=True, num_workers=self.opt.workers)

    return dataloader

  def getBirdsnap(self):
    return data.DataLoader(BirdsnapDataset(self.opt.matfile), batch_size=self.opt.batchSize, shuffle=True, num_workers=self.opt.workers)

  def getMnist(self):
    try:
      dataroot = self.getPath('mnist')
    except KeyError:
      print "Dataroot for mnist not found"
      sys.exit()
    return data.DataLoader(generate_mnist_distribution(datadir=dataroot, probs=self.opt.proportions), batch_size=self.opt.batchSize, shuffle=True, num_workers=self.opt.workers)


def main():
  loader = BirdsnapDataset(matfile)
  v = loader[0]
  print type(v)
  print type(np.array(v))
  print np.max(np.array(v))
  print np.min(np.array(v))
  print v

if __name__=='__main__':
  main()

