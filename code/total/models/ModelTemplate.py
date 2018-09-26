from mlworkflow import Loader

class ModelTemplate(Loader):
  def __init__(self, config, args):
    super(ModelTemplate, self).__init__(config, args)

  # Give an error if loaderTemplateObj cannot be made compatible
  def train(self, loaderTemplateObj, nepochs):
    raise NotImplementedError()

  # Get a set of images, returned as a tensor
  def sample(self, nSamples):
    raise NotImplementedError()

  # Get a dict with probabilities, codes, deep features, etc.
  def probSample(self, nSamples, deepFeatures=None):
    raise NotImplementedError()

  # Report performance on test loader
  def test(self, loaderTemplateObj):
    raise NotImplementedError()

  # Save model using proper keywords
  def saveCheckpoint(self):
    raise NotImplementedError()

  # Load self if args say so
    # can't call this load because it conflicts with the other load function
  def loadModel(self):
    raise NotImplementedError()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
