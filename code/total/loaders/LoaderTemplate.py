from copy import copy
from mlmanager import Loader

class LoaderTemplate(Loader):
  def __init__(self, config, args):
    super(LoaderTemplate, self).__init__(config,args)
    self.opt = {
      'batchSize':64,
      'workers':2,
      'shuffle':True
    }
    self.opt.update(copy(args))

    self.batchSize = self.opt['batchSize']
    self.workers = self.opt['workers']
    self.shuffle = self.opt['shuffle']

  # Return the torch dataset object for the dataset
  # Outshape is the desired tensor shape
  # Everything will be normalized between -1 and 1
  # returnClass decides whether the dataset object will
  #  return a pair (x,y) or just x
  # This function should give an error if outShape is 
  #  not compatible with the dataset
  def getDataset(self, outShape=None, mode='train', returnLabel=False):
    raise NotImplementedError()

  # Return the torch dataloader object for the dataset
  def getDataloader(self, outShape=None, mode='train', returnLabel=False):
    raise NotImplementedError()


