from mlworkflow import Loader
from easydict import EasyDict as edict

import torch.utils.data as data
#from os.path import join
from scipy.io import loadmat
import numpy as np
import torch

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

class DataloaderRyen(Loader):
  def __init__(self, config, args):
    super(DataloaderRyen, self).__init__(config, args)
    opt = {
      'matfile':'/vulcan/scratch/krusinga/birdsnap/birdsnap/download/birdsnap.mat',
      'batchSize':64,
      'workers':2
    }
    opt.update(args)
    self.opt = edict(opt)

  def getLoader(self):
    return data.DataLoader(BirdsnapDataset(self.opt.matfile), batch_size=self.opt.batchSize, shuffle=True, num_workers=self.opt.workers)


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

