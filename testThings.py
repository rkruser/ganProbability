import torch
from torch.autograd import Variable

from loaders import getLoaders
from models import getModels
from train import totalAccuracy


def test1():
  D10 = getModels('densenet')[0]
  D10.load_state_dict(torch.load('generated/final/densenet_cifar/netEmb_14.pth'))
  D10 = D10.cuda()
  cifar10 = getLoaders('cifar10',returnLabel=True, mode='test')
  print totalAccuracy(D10, cifar10, cuda=True)

def test2():
  D10 = getModels('densenet')[0]
  D10.load_state_dict(torch.load('generated/final/densenet_mnist/netEmb_10.pth'))
  D10 = D10.cuda()
  mnist = getLoaders('mnist',returnLabel=True, mode='test')
  print totalAccuracy(D10, mnist, cuda=True)



if __name__=='__main__':
  test1()
