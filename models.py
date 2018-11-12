# Models
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F #weird
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            init.constant(m.bias,0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.normal(m.weight, std=1e-2)
        # init.orthogonal(m.weight)
        #init.xavier_uniform(m.weight, gain=1.4)
        if m.bias is not None:
            init.constant(m.bias, 0.0)

# Wrap a neural net and only return its first return value
class NthArgWrapper(nn.Module):
    def __init__(self, net, arg):
        super(NthArgWrapper, self).__init__()
        self.net = net
        if hasattr(self.net, 'outClasses'):
            self.outClasses = self.net.outClasses
        self.arg = arg

    def numOutDims(self):
        return self.net.numOutDims(self.arg)

    def outshape(self):
        return self.net.outshape(self.arg)

    def numLatent(self):
        return self.net.numLatent()

    def imsize(self):
        return self.net.imsize()

    def numColors(self):
        return self.net.imsize()

    def numOutClasses(self):
        return self.net.numOutClasses()

    def setArg(self, arg):
        self.arg = arg


    def forward(self, x):
        result = self.net(x)
        return result[self.arg]

class DeepFeaturesWrapper(nn.Module):
    def __init__(self,netG, netEmb):
        super(DeepFeaturesWrapper,self).__init__()
        self.netG = netG
        self.netEmb = netEmb

    def numLatent(self):
        return self.netG.numLatent()

    def numOutDims(self):
        return self.netEmb.numOutDims()

    def outshape(self):
        return self.netEmb.outshape()

    def imsize(self):
        return None

    def numColors(self):
        return None

    def forward(self, x):
        return self.netEmb(self.netG(x))


class NetGDeep(nn.Module):
    def __init__(self, nz=100, ngf=625, ndeep=384):
        super(NetGDeep, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.ndeep = ndeep
        self.main = nn.Sequential(
                nn.Linear(nz, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ndeep),
               # nn.ReLU(inplace=True) # Because the deep features used have a ReLU
            )

    def numLatent(self):
        return self.nz

    def outshape(self):
        return [self.ndeep]

    def imsize(self):
        return self.ndeep

    def numOutDims(self):
        return self.ndeep

    def numColors(self):
        return None

    def forward(self, x):
        return self.main(x)

class NetDDeep(nn.Module):
    def __init__(self, ngf=625, ndeep=384):
        super(NetDDeep, self).__init__()
        self.ngf = ngf
        self.ndeep = ndeep
        self.main = nn.Sequential(
                nn.Linear(ndeep, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, 1)                             
            )

    def numLatent(self):
        return None

    def outshape(self):
        return [1]

    def imsize(self):
        return self.ndeep

    def numOutDims(self):
        return 1

    def numColors(self):
        return None

    def forward(self, x):
        return self.main(x).view(-1,1).squeeze(1)


class NetGDeepV2(nn.Module):
    def __init__(self, nz=100, ngf=128, ndeep=6144):
        super(NetGDeepV2, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.ndeep = ndeep
        self.main = nn.Sequential(
                nn.Linear(nz, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, 2*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(2*ngf, 2*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(2*ngf, 4*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(4*ngf, 4*ngf),
               # nn.ReLU(inplace=True),
               # nn.Linear(4*ngf, 8*ngf),
               # nn.ReLU(8*ngf),
               # nn.Linear(8*ngf, 8*ngf),
               # nn.ReLU(8*ngf),
               # nn.Linear(8*ngf,16*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(4*ngf, ndeep),
               # nn.ReLU(inplace=True) # Because the deep features used have a ReLU
            )

    def numLatent(self):
        return self.nz

    def outshape(self):
        return [self.ndeep]

    def imsize(self):
        return self.ndeep

    def numOutDims(self):
        return self.ndeep

    def numColors(self):
        return None

    def forward(self, x):
        return self.main(x)

class NetDDeepV2(nn.Module):
    def __init__(self, nz=100, ngf=128, ndeep=6144):
        super(NetDDeepV2, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.ndeep = ndeep
        self.main = nn.Sequential(
                nn.Linear(ndeep, 4*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(4*ngf, 4*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(4*ngf, 2*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(2*ngf, 2*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(2*ngf, ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, ngf),
               # nn.ReLU(inplace=True),
               # nn.Linear(4*ngf, 8*ngf),
               # nn.ReLU(8*ngf),
               # nn.Linear(8*ngf, 8*ngf),
               # nn.ReLU(8*ngf),
               # nn.Linear(8*ngf,16*ngf),
                nn.ReLU(inplace=True),
                nn.Linear(ngf, 1),
               # nn.ReLU(inplace=True) # Because the deep features used have a ReLU
            )

    def numLatent(self):
        return None

    def outshape(self):
        return [1]

    def imsize(self):
        return self.ndeep

    def numOutDims(self):
        return 1

    def numColors(self):
        return None

    def forward(self, x):
        return self.main(x).view(-1,1).squeeze(1)


# Model for 32 by 32 images
class NetG32(nn.Module):
  def __init__(self, nz=100, ngf=64, nc=3):
    super(NetG32, self).__init__()
    self.nz = nz
    self.ngf = ngf
    self.nc = nc
    self.imsize=32
    self.main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 4 x 4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 7 x 7
        nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
        # for 28 x 28
       # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 14 x 14
        nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 32 x 32
    )

  def numLatent(self):
    return self.nz

  def outshape(self):
    return [self.nc, self.imsize, self.imsize]

  def imsize(self):
    return self.imsize

  def numOutDims(self):
    return self.nc*self.imsize*self.imsize

  def numColors(self):
    return self.nc

  def forward(self, input):
   # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
   #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
   # else:
    input = input.unsqueeze(2).unsqueeze(3)
    output = self.main(input)
    return output


class NetD32(nn.Module):
  def __init__(self, nz=100, ndf=64, nc=3, infogan=False):
    super(NetD32, self).__init__()
    self.nz = nz
    self.infogan = infogan
    self.main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 16 x 16
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        # for 28 x 28
       # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
    )

    # Predictor takes main output and produces a probability
    self.predictor = nn.Sequential(
        nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False)
       # nn.Sigmoid()
        # Output is a single scalar
    )

    # Output size is nz
    if self.infogan:
      self.reconstructor = nn.Conv2d(ndf*4, nz, 4, 1, 0)

  def numOutDims(self):
    return 1

  def outshape(self):
    return [1]



  def forward(self, x):
    # Removed the x.data part in this and in netG
   # if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
   #   output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
   # else:
    output = self.main(x)

    prediction = self.predictor(output).view(-1,1).squeeze(1)

    if self.infogan:
      reconstruction = self.reconstructor(output)
      return prediction, reconstruction.view(-1,self.nz) # output is (batchSize x nz) for recon
    else:
      return prediction


# Regressor
class NetP32(nn.Module):
    def __init__(self, nc, npf):
        super(NetP32, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, npf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(npf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf) x 16 x 16
            nn.Conv2d(npf, npf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(npf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf*2) x 8 x 8
            nn.Conv2d(npf * 2, npf * 4, 4, 2, 1, bias=True),
            # for 28 x 28
           # nn.Conv2d(npf * 2, npf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(npf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf*4) x 4 x 4
            nn.Conv2d(npf * 4, 1, 4, 1, 0, bias=True),
            # nn.ReLU(inplace=True)
        )

    def numOutDims(self):
        return 1

    def outshape(self):
        return [1]


    def forward(self, input):
       # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
       #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
       # else:
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# What is the best shape for this net?
# This predicts log probabilities
class NetPLatent(nn.Module):
  def __init__(self, nz=100, npf=64):
    super(NetPLatent,self).__init__()
    self.main = nn.Sequential(
      nn.Linear(nz, npf*4),
      nn.ReLU(inplace=True),
      nn.Linear(npf*4, npf*2),
      nn.ReLU(inplace=True),
      nn.Linear(npf*2, npf),
      nn.ReLU(inplace=True),
      nn.Linear(npf, 1) # no log probs
    )

  def numOutDims(self):
    return 1

  def outshape(self):
    return [1]


  def forward(self, x):
    output = self.main(x)
      # output = -torch.abs(output)+th  # Why do this?

    return output.view(-1,1).squeeze(1) 


# Every model for MoG - Generator
class mog_netG(nn.Module):
    def __init__(self, nz):
        super(mog_netG, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz,256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def numLatent(self):
        return self.nz

    def outshape(self):
        return [2]

    def imsize(self):
      return None

    def numOutDims(self):
      return 2

    def numColors(self):
      return None


    def forward(self, input):
      output = self.main(input)
      return output

# Discriminator
class mog_netD(nn.Module):
    def __init__(self):
        super(mog_netD, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
           # nn.Sigmoid()
        )

    def numOutDims(self):
        return 1

    def outshape(self):
        return [1]


    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)


# Lenet model for extracting features
class Lenet32(nn.Module):
    def __init__(self, nc):
        super(Lenet32, self).__init__()
        self.nc = nc
        self.outClasses = 10
        self.features = nn.Sequential(
            nn.Conv2d(nc, 20, 5, 1, bias=True),
            # nn.MaxPool2d(2,2),
            nn.AvgPool2d(2,2),
            nn.Conv2d(20, 50, 5, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.AvgPool2d(2, 2),
        )
        self.features2 = nn.Linear(5*5*50, 500)
        self.main = nn.Sequential(
            nn.Linear(500, 10)
            )

    def numOutClasses(self):
      return self.outClasses

    def outshape(self, arg):
        if arg == 0:
          return [500]
        elif arg == 1:
          return [10]
        else:
          raise ValueError(str(self)+"\nArgument index out of range")

    def imsize(self):
      return 32

    def numOutDims(self, arg):
      if arg == 0:
        return 500
      elif arg == 1:
        return 10
      else:
        raise ValueError(str(self)+"\nArgument index out of range")

    def numColors(self):
      return self.nc


    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
        #     output = output.view(input.size(0), -1)
        #     output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
        #     output1 = self.main(output)
        # else:
        output = self.features(input)
        output = output.view(input.size(0), -1)
        output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
        output1 = self.main(output)

        return output, output1


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, cuda=False):
        super(RealNVP, self).__init__()
        self.nz = 2
        self.hascuda = cuda
       # self.logPrior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        # Perhaps have a scale parameter here, but ntic
        # *** Must factor scale param into the jacobian
       # self.scale = nn.Parameter(torch.FloatTensor(1))
        
    # latent codes to data
    def forward(self, z):
        x = z
        log_det_J = Variable(torch.zeros(x.size(0)))
        if self.hascuda:
            log_det_J = log_det_J.cuda()
        for i in range(len(self.t)):
            x_ = x*self.mask[i].expand_as(x)
            s = self.s[i](x_)*(1 - self.mask[i]).expand_as(x)
            t = self.t[i](x_)*(1 - self.mask[i]).expand_as(x)
            x = x_ + (1 - self.mask[i]).expand_as(x) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        logProb = self.priorLogProb(z)-log_det_J
        return x, logProb

    # Data to latent codes
    def invert(self, x):
       # log_det_J, z = x.new_zeros(x.shape[0]), x
        log_det_J = Variable(torch.zeros(x.size(0)))
        if self.hascuda:
            log_det_J = log_det_J.cuda()
        z = x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i].expand_as(z) * z
            s = self.s[i](z_) * (1-self.mask[i]).expand_as(z)
            t = self.t[i](z_) * (1-self.mask[i]).expand_as(z)
            z = (1 - self.mask[i]).expand_as(z) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        logProb = self.priorLogProb(z)+log_det_J
        return z, logProb
    
   # def log_prob(self,x):
   #     z, logp = self.f(x)
       # return self.priorLogPrior(z) + logp

    def priorLogProb(self, z):
        c = float(-2*np.log(np.sqrt(2*np.pi)))
        return c-0.5*(z**2).sum(dim=1)

    def numLatent(self):
        return self.nz


from densenet import densenet_cifar

# returnFeats and returnClf are for embeddings
def getModels(model, nc=3, imsize=32, hidden=64, ndeephidden=625, nz=100, cuda=False, returnFeats=False):
    if model == 'dcgan':
    	return [NetG32(nc=nc, ngf=hidden, nz=nz), NetD32(nc=nc, ndf=hidden, nz=nz)]
    elif model == 'DeepGAN384':
        return [NetGDeep(nz=nz, ngf=ndeephidden, ndeep=384), NetDDeep(ngf=ndeephidden, ndeep=384)]
    elif model == 'DeepGAN10':
        return [NetGDeep(nz=nz, ngf=ndeephidden, ndeep=10), NetDDeep(ngf=ndeephidden, ndeep=10)]
    elif model == 'pixelRegressor':
    	return [NetP32(nc=nc, npf=hidden)]
    elif model == 'DeepRegressor10':
        return [NetDDeep(ngf=ndeephidden, ndeep=10)]
    elif model == 'DeepRegressor384':
        return [NetDDeep(ngf=ndeephidden, ndeep=384)]
    elif model == 'lenetEmbedding':
    	return [NthArgWrapper(Lenet32(nc=nc), 1)]
    elif model == 'densenet':
        model = densenet_cifar()
        if returnFeats:
            return [NthArgWrapper(model, 0)]
        else:
            return [NthArgWrapper(model, 1)]

    elif model == 'mogNVP':
        nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        #prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        flow = RealNVP(nets, nett, masks, cuda)
        return [flow, mog_netD()]
    elif model == 'pureNVP':
        nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        #prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        flow = RealNVP(nets, nett, masks, cuda)
        return [flow]
    elif model == 'mog':
        return [mog_netG(nz), mog_netD()]
    else:
        raise NameError("No such model")






# Regressor - fc2 features
#class NetF10(nn.Module):
#    def __init__(self, ngpu, nc=None, npf=None):
#        super(NetF10, self).__init__()
#        self.ngpu = ngpu
#        self.main = nn.Sequential(
#            nn.Linear(10, 32),
#            nn.ReLU(inplace=True),
#            nn.Linear(32, 32),
#            nn.ReLU(inplace=True),
#            nn.Linear(32, 64),
#            nn.ReLU(inplace=True),
#            nn.Linear(64, 128),
#            nn.ReLU(inplace=True),
#            nn.Linear(128, 256),
#            nn.ReLU(inplace=True),
#            nn.Linear(256,1)
#        )
#
#    def forward(self, input):
#        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#        else:
#            output = self.main(input)
#
#        return output.view(-1, 1).squeeze(1)
#
#
## Regressor - fc1 features
#class NetF500(nn.Module):
#    def __init__(self, ngpu, nc=None, npf=None):
#        super(NetF500, self).__init__()
#        self.ngpu = ngpu
#        self.main = nn.Sequential(
#            nn.Linear(500,800),
#            nn.ReLU(inplace=True),
#            nn.Linear(800,500),
#            nn.ReLU(inplace=True),
#            nn.Linear(500,1)
#        )
#
#    def forward(self, input):
#        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#        else:
#            output = self.main(input)
#
#        return output.view(-1, 1).squeeze(1)
#
#class DeepRegressor(nn.Module):
#    def __init__(self, featureExtractor, nOutFeatures=500, ngpu=0):
#        super(DeepRegressor, self).__init__()
#        self.deepFeatures = featureExtractor
#        for m in self.deepFeatures.parameters():
#            m.requires_grad = False
#
#        self.ngpu = ngpu
#        self.nOutFeatures = nOutFeatures
#        self.main = nn.Sequential(
#            nn.Linear(self.nOutFeatures, 256),
#            nn.ReLU(inplace=True),
#            nn.Linear(256, 128),
#            nn.ReLU(inplace=True),
#            nn.Linear(128, 64),
#            nn.ReLU(inplace=True),
#            nn.Linear(64, 32),
#            nn.ReLU(inplace=True),
#            nn.Linear(32, 16),
#            nn.ReLU(inplace=True),
#            nn.Linear(16,1)
#        )
#        self.main.apply(weights_init)
#
#    def numOutDims(self):
#        return 1
#
#    def outshape(self):
#        return [1]
#
#
#    def forward(self, x):
#        feats, _ = self.deepFeatures(x)
#        output = self.main(feats)
#            # output -= 8*F.relu(output - th)
#           # output = -torch.abs(output) + th
#
#        return output.view(-1, 1).squeeze(1)
#
