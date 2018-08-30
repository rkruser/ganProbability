from mlworkflow import Loader
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.normal(m.weight, std=1e-2)
        # init.orthogonal(m.weight)
        #init.xavier_uniform(m.weight, gain=1.4)
        if m.bias is not None:
            init.constant(m.bias, 0.0)

# Model for 28 by 28 images
class NetG28(nn.Module):
  def __init__(self, nz=100, ngf=64, nc=1, ngpu=0):
    super(NetG28, self).__init__()
    self.ngpu = ngpu
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
        # nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
        # for 28 x 28
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 14 x 14
        nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 28 x 28
    )

  def forward(self, input):
    if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
        output = self.main(input)
    return output


class NetG64(nn.Module):
  def __init__(self, nz=100, ngf=64, nc=3, ngpu=0):
    super(NetG64,self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # Outsize = stride x (inputSize-1) + kernel - 2*padding
        # e.g. 1*(1-1)+4-2*0 = 4
        nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), #Why no bias?
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True), # inplace=True
        # State: (ngf x 8) x 4 x 4
        nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True),
        # State: (ngf x 4) x 8 x 8
        nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True),
        # State: (ngf x 2) x 16 x 16
        nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # State: (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # (nc) x 64 x 64
      )

  def forward(self, x):
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)
    return output

# Discriminator
class NetD28(nn.Module):
  def __init__(self, nz=100, ndf=64, nc=1, ngpu=0):
    super(NetD28, self).__init__()
    self.ngpu = ngpu
    self.nz = nz
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
        # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        # for 28 x 28
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
    )

    # Predictor takes main output and produces a probability
    self.predictor = nn.Sequential(
        nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
        # Output is a single scalar
    )

    # Output size is nz
    self.reconstructor = nn.Conv2d(ndf*4, nz, 4, 1, 0)


  def forward(self, x):
    # Removed the x.data part in this and in netG
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)

    prediction = self.predictor(output)
    reconstruction = self.reconstructor(output)
    return prediction.view(-1,1).squeeze(1), reconstruction.view(-1,self.nz) # output is (batchSize x nz) for recon


class NetD64(nn.Module):
  def __init__(self, nz=100, ndf=64, nc=3, ngpu=0):
    super(NetD64,self).__init__()
    self.ngpu = ngpu
    self.nz = nz
    self.main = nn.Sequential(
        # input (nc) x 64 x 64
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # New state: floor((inputSize - kernel + 2*padding) / stride) + 1
        # (ndf) x 32 x 32
        nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf*2),
        nn.LeakyReLU(0.2, inplace=True),
        # (ndf*2) x 16 x 16
        nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        # (ndf*4) x 8 x 8
        nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2, inplace=True)
      )
      # Output of main is (ndf*4) x 4 x 4

    # Predictor takes main output and produces a probability
    self.predictor = nn.Sequential(
        nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
        # Output is a single scalar
    )

    # Output size is nz
    self.reconstructor = nn.Conv2d(ndf*8, nz, 4, 1, 0)

  def forward(self, x):
    # Removed the x.data part in this and in netG
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)

    prediction = self.predictor(output)
    reconstruction = self.reconstructor(output)
    return prediction.view(-1,1).squeeze(1), reconstruction.view(-1,self.nz) # output is (batchSize x nz) for recon


# What is the best shape for this net?
# This predicts log probabilities
class NetPLatent(nn.Module):
  def __init__(self, nz=100, npf=64, ngpu=0):
    super(NetPLatent,self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      nn.Linear(nz, npf*4),
      nn.ReLU(inplace=True),
      nn.Linear(npf*4, npf*2),
      nn.ReLU(inplace=True),
      nn.Linear(npf*2, npf),
      nn.ReLU(inplace=True),
      nn.Linear(npf, 1) # no log probs
    )

  def forward(self, x):
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)
      # output = -torch.abs(output)+th  # Why do this?

    return output.view(-1,1).squeeze(1) 

class ModelLoaderRyen(Loader):
  def __init__(self, config, args):
    super(ModelLoaderRyen,self).__init__(config, args)
    opt = {
      'dataset':'birdsnap',
      'nz':100,
      'ngf':64,
      'ndf':64,
      'npf':64,
      'nc':3,
      'ngpu':0,
      'cuda':False,
      'manualSeed':None,
      'lr':0.0002,
      'beta1':0.5,
      'reconScale':0.5
    }
    opt.update(args)
    self.opt = edict(opt)

  def getModel(self):
    modelName = self.opt.dataset
    if modelName == 'birdsnap':
      return self.getBirdsnap()
    elif modelName == 'mnist':
      return self.getMnist()
    else:
      raise ValueError("Unknown model %s"%modelName)

  def getGenerator(self):
    modelName = self.opt.dataset
    if modelName == 'birdsnap':
      netG = NetG64(self.opt.nz,self.opt.ngf,self.opt.nc,self.opt.ngpu)
    elif modelName == 'mnist':
      netG = NetG28(self.opt.nz,self.opt.ngf,self.opt.nc,self.opt.ngpu)
    else:
      raise ValueError("Unknown model %s"%modelName)
    netG.apply(weights_init)
    if self.opt.netG != '':
      self.log("Loading netG from file")
      netG.load_state_dict(self.files.load(self.opt.netG, instance=self.opt.netGinstance,
            number=self.opt.netGexpNum,loader='torch'))
    return netG

  def getDiscriminator(self):
    modelName = self.opt.dataset
    if modelName == 'birdsnap':
      netD = NetD64(self.opt.nz,self.opt.ndf,self.opt.nc,self.opt.ngpu)
    elif modelName == 'mnist':
      netD = NetD28(self.opt.nz,self.opt.ndf,self.opt.nc,self.opt.ngpu)
    else:
      raise ValueError("Unknown model %s"%modelName)
    netD.apply(weights_init)
    if self.opt.netD != '':
      self.log("Loading netD from file")
      netD.load_state_dict(self.files.load(self.opt.netD, instance=self.opt.netDinstance,
            number=self.opt.netDexpNum,loader='torch'))
    return netD


  def getMnist(self):
    return self.getGenericProblem(NetG28, NetD28)
    
  def getBirdsnap(self):
    return self.getGenericProblem(NetG64, NetD64)

  def getGenericProblem(self, NETWORK_G, NETWORK_D):
    netG = NETWORK_G(self.opt.nz,self.opt.ngf,self.opt.nc,self.opt.ngpu)
    netD = NETWORK_D(self.opt.nz,self.opt.ndf,self.opt.nc,self.opt.ngpu)
    model = {
      'netG':netG,
      'netD':netD,
      'cuda':self.opt.cuda,
      'criterion':nn.BCELoss(), #what about mse?
      'reconstructionLoss':nn.MSELoss(),
      'reconScale':self.opt.reconScale,
      'nz':self.opt.nz,
      'optimizerG':optim.Adam(netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1,0.999)),
      'optimizerD':optim.Adam(netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1,0.999))
    }
    model = edict(model)

    if self.opt.manualSeed is None:
      manualSeed = random.randint(1,10000)
      self.log("Using random seed %d"%manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if self.opt.cuda:
      torch.cuda.manual_seed_all(manualSeed)
    cudnn.benchmark = True 
    
    # possibly load from file, else
    model.netG.apply(weights_init)
    model.netD.apply(weights_init)

    if self.opt.cuda:
      model.netG.cuda()
      model.netD.cuda()
      model.criterion.cuda()

    print model
      
    return model


