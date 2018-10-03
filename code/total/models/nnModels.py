from mlworkflow import Loader
from easydict import EasyDict as edict
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F #weird
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
import torch.nn.init as init

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


# Model for 32 by 32 images
class NetG32(nn.Module):
  def __init__(self, nz=100, ngf=64, nc=1, ngpu=0):
    super(NetG32, self).__init__()
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
        nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
        # for 28 x 28
#        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 14 x 14
        nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 32 x 32
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
  def __init__(self, nz=100, ndf=64, nc=1, ngpu=0, infogan=False):
    super(NetD28, self).__init__()
    self.ngpu = ngpu
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
    if self.infogan:
      self.reconstructor = nn.Conv2d(ndf*4, nz, 4, 1, 0)


  def forward(self, x):
    # Removed the x.data part in this and in netG
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)

    prediction = self.predictor(output).view(-1,1).squeeze(1)

    if self.infogan:
      reconstruction = self.reconstructor(output)
      return prediction, reconstruction.view(-1,self.nz) # output is (batchSize x nz) for recon
    else:
      return prediction


class NetD32(nn.Module):
  def __init__(self, nz=100, ndf=64, nc=1, ngpu=0, infogan=False):
    super(NetD32, self).__init__()
    self.ngpu = ngpu
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
        nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
        # Output is a single scalar
    )

    # Output size is nz
    if self.infogan:
      self.reconstructor = nn.Conv2d(ndf*4, nz, 4, 1, 0)


  def forward(self, x):
    # Removed the x.data part in this and in netG
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)

    prediction = self.predictor(output).view(-1,1).squeeze(1)

    if self.infogan:
      reconstruction = self.reconstructor(output)
      return prediction, reconstruction.view(-1,self.nz) # output is (batchSize x nz) for recon
    else:
      return prediction


class NetD64(nn.Module):
  def __init__(self, nz=100, ndf=64, nc=3, ngpu=0, infogan=False):
    super(NetD64,self).__init__()
    self.ngpu = ngpu
    self.nz = nz
    self.infogan = infogan
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
    if self.infogan:
      self.reconstructor = nn.Conv2d(ndf*8, nz, 4, 1, 0)

  def forward(self, x):
    # Removed the x.data part in this and in netG
    if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
    else:
      output = self.main(x)

    prediction = self.predictor(output).view(-1,1).squeeze(1)

    if self.infogan:
      reconstruction = self.reconstructor(output)
      return prediction, reconstruction.view(-1,self.nz) # output is (batchSize x nz) for recon
    else:
      return prediction

# ===========================

# Regressor
class NetP28(nn.Module):
    def __init__(self, ngpu, nc, npf):
        super(NetP28, self).__init__()
        self.ngpu = ngpu
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
            # nn.Conv2d(npf * 2, npf * 4, 4, 2, 1, bias=True),
            # for 28 x 28
            nn.Conv2d(npf * 2, npf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(npf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf*4) x 4 x 4
            nn.Conv2d(npf * 4, 1, 4, 1, 0, bias=True),
            # nn.ReLU(inplace=True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# Regressor
class NetP32(nn.Module):
    def __init__(self, ngpu, nc, npf):
        super(NetP32, self).__init__()
        self.ngpu = ngpu
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
#            nn.Conv2d(npf * 2, npf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(npf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf*4) x 4 x 4
            nn.Conv2d(npf * 4, 1, 4, 1, 0, bias=True),
            # nn.ReLU(inplace=True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# Regressor
class NetP64(nn.Module):
    def __init__(self, ngpu, nc, npf):
        super(NetP64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, npf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(npf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(npf, npf*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(npf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf) x 16 x 16
            nn.Conv2d(npf*2, npf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(npf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf*2) x 8 x 8
            nn.Conv2d(npf * 4, npf * 8, 4, 2, 1, bias=True),
            # for 28 x 28
#            nn.Conv2d(npf * 2, npf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(npf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (npf*4) x 4 x 4
            nn.Conv2d(npf * 8, 1, 4, 1, 0, bias=True),
            # nn.ReLU(inplace=True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)



## Regressor - conv1 features
# class _netF(nn.Module):
#     def __init__(self, ngpu):
#         super(_netF, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Linear(1250,800),
#             nn.ReLU(inplace=True),
#             nn.Linear(800,500),
#             nn.ReLU(inplace=True),
#             nn.Linear(500,1)
#         )
#
#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#
#         return output.view(-1, 1).squeeze(1)

## Regressor - fc1 features
# class _netF(nn.Module):
#     def __init__(self, ngpu):
#         super(_netF, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Linear(500,800),
#             nn.ReLU(inplace=True),
#             nn.Linear(800,500),
#             nn.ReLU(inplace=True),
#             nn.Linear(500,1)
#         )
#
#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#
#         return output.view(-1, 1).squeeze(1)

# Regressor - fc2 features
class _netF(nn.Module):
    def __init__(self, ngpu):
        super(_netF, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1)
        )

    def forward(self, input, th):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            # output -= 8*F.relu(output - th)
            output = -torch.abs(output) + th

        return output.view(-1, 1).squeeze(1)

# Lenet model for extracting features
class _lenet(nn.Module):
    def __init__(self, ngpu):
        super(_lenet, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1, bias=True),
            # nn.MaxPool2d(2,2),
            nn.AvgPool2d(2,2),
            nn.Conv2d(20, 50, 5, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.AvgPool2d(2, 2),
        )
        self.features2 = nn.Linear(5*5*50, 500)
        self.main = nn.Linear(500, 10)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)
        else:
            output = self.features(input)
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)

        return output, output1

# Classifier for which Adversarial exmaples were generated
class _lenet_ad(nn.Module):
    def __init__(self, ngpu):
        super(_lenet_ad, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(784, 625),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(625, 625),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(625, 10)
        )

    def forward(self, input):
        output = input.view(input.size(0), -1)
        output = self.features(output)
        return output, output

# Every model for MoG - Generator
class mog_netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(mog_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz,128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
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
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)




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


######### Deep Features #########

# Lenet model for extracting features
class Lenet32(nn.Module):
    def __init__(self, ngpu, nc):
        super(Lenet32, self).__init__()
        self.ngpu = ngpu
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
            nn.Linear(500, 10),
            nn.Softmax() #Change in others as well
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)
        else:
            output = self.features(input)
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)

        return output, output1

# Lenet model for extracting features
class Lenet28(nn.Module):
    def __init__(self, ngpu, nc):
        super(Lenet28, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            nn.Conv2d(nc, 20, 5, 1, bias=True),
            # 32 x 32 out
            nn.AvgPool2d(2,2), #16 x 16
            nn.Conv2d(20, 50, 5, 1, bias=True), #16 x 16
            # nn.MaxPool2d(2, 2),
            nn.AvgPool2d(2, 2),
        )
        self.features2 = nn.Linear(4*4*50, 500)
        self.main = nn.Sequential(
            nn.Linear(500, 10),
            nn.Softmax() #Change in others as well
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)
        else:
            output = self.features(input)
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)

        return output, output1

# Lenet model for extracting features
class Lenet64(nn.Module):
    def __init__(self, ngpu, nc):
        super(Lenet64, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            # 64 x 64
            nn.Conv2d(nc, 20, 5, 1, 2, bias=True), # 64x64
            nn.AvgPool2d(2,2), #32 x 32

            nn.Conv2d(20, 50, 5, 1, bias=True),
            # 32 x 32 out
            nn.AvgPool2d(2,2), #16 x 16
            nn.Conv2d(50, 100, 5, 1, bias=True), #16 x 16
            # nn.MaxPool2d(2, 2),
            nn.AvgPool2d(2, 2),
        )
        self.features2 = nn.Linear(5*5*100, 500)
        self.main = nn.Sequential(
            nn.Linear(500, 10),
            nn.Softmax() #Change in others as well
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)
        else:
            output = self.features(input)
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)

        return output, output1

# Lenet model for extracting features
class Lenet128(nn.Module):
    def __init__(self, ngpu, nc):
        super(Lenet128, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            # 128 x 128
            nn.Conv2d(nc, 20, 5, 1, 2, bias=True), # 128 x 128
            nn.AvgPool2d(2,2), #64 x 64

            # 64 x 64
            nn.Conv2d(20, 50, 5, 1, 2, bias=True), # 64x64
            nn.AvgPool2d(2,2), #32 x 32

            nn.Conv2d(50, 100, 5, 1, bias=True),
            # 28 x 28 out
            nn.AvgPool2d(2,2), #14 x 14
            nn.Conv2d(100, 200, 5, 1, bias=True), #10 x 10
            nn.AvgPool2d(2, 2), #5 x 5 
        )
        self.features2 = nn.Linear(5*5*200, 500)
        self.main = nn.Sequential(
            nn.Linear(500, 10),
            nn.Softmax() #Change in others as well
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.features, input, range(self.ngpu))
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)
        else:
            output = self.features(input)
            output = output.view(input.size(0), -1)
            output = F.leaky_relu(self.features2(output), negative_slope=0.2, inplace=True)
            output1 = self.main(output)

        return output, output1



class DeepRegressor(nn.Module):
    def __init__(self, featureExtractor, nOutFeatures=500, ngpu=0):
        super(DeepRegressor, self).__init__()
        self.deepFeatures = featureExtractor
        for m in self.deepFeatures.parameters():
            m.requires_grad = False

        self.ngpu = ngpu
        self.nOutFeatures = nOutFeatures
        self.main = nn.Sequential(
            nn.Linear(self.nOutFeatures, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16,1)
        )
        self.main.apply(weights_init)

    def forward(self, x):
        feats, _ = self.deepFeatures(x)
        if isinstance(feats.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, feats, range(self.ngpu))
        else:
            output = self.main(feats)
            # output -= 8*F.relu(output - th)
#            output = -torch.abs(output) + th

        return output.view(-1, 1).squeeze(1)
       



