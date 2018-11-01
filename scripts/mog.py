import os.path as osp
import torch
import torch.nn as nn
import numpy as np


#import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline

#from pylab import rcParams
#rcParams['figure.figsize'] = 10, 8
#rcParams['figure.dpi'] = 300

#import torch
#from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

#from sklearn import cluster, datasets, mixture
#from sklearn.preprocessing import StandardScaler

# https://github.com/ars-ashuha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb

def getMogMeans(nmeans=8, scale=1.0, omit=None):
  mogMeans = scale*np.array([[np.cos(i*2*np.pi/nmeans), np.sin(i*2*np.pi/nmeans)] for i in range(nmeans) if i != omit]).astype(np.float32)
  return mogMeans


def getRandomMeans(mogMeans, npts):
  indices = np.random.choice(len(mogMeans), npts, replace=True)
  return torch.from_numpy(mogMeans[indices])


eightMeans = getMogMeans()
eightMinusOne = getMogMeans(omit=1)

def mogData(n, mogMeans=eightMeans, stdev=0.01):
  # Get n samples of a mog circle

  data = torch.Tensor(n,2).normal_(0,stdev)+getRandomMeans(mogMeans,n)
  return data



class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.logPrior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.logPrior(z) + logp
        
#    def sample(self, batchSize): 
#        z = self.prior.sample((batchSize, 1))
#        logp = self.prior.log_prob(z)
#        x = self.g(z)
#        return x


def RunRealNVP():
  nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
  nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
  masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
  #prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
  def priorLogProb(z):
    c = -z.size(1)*np.log(np.sqrt(2*np.pi))
    return c-0.5*(z**2).sum(dim=1)
  flow = RealNVP(nets, nett, masks, priorLogProb)

  optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)
#  for t in range(5001):    
#    noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)
#    loss = -flow.log_prob(torch.from_numpy(noisy_moons)).mean()
#        
#    optimizer.zero_grad()
#    loss.backward(retain_graph=True)
#    optimizer.step()
#        
#    if t % 500 == 0:
#      print('iter %s:' % t, 'loss = %.3f' % loss)



class mog_netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(mog_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz,128),
#            nn.Tanh(),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
#            nn.Tanh(),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# Discriminator
class mog_netD(nn.Module):
    def __init__(self, ngpu):
        super(mog_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, 128),
#            nn.Tanh(),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
#            nn.Tanh(),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

