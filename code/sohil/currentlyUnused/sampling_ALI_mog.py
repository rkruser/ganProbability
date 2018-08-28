from __future__ import print_function
import argparse
import os
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import time
from torch.autograd import Variable
from models import generate_data_uniform


parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--encdim', type=int, default=200, help='hidden dimension of encoder')
parser.add_argument('--gendim', type=int, default=200, help='hidden dimension of generator')
parser.add_argument('--samples', type=int, default=100000)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to load model)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--forward', help='Computes Jacobian using Forward pass', action='store_true')
parser.add_argument('--gen', help='Computes Jacobian using Generator network', action='store_true')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
samples = int(opt.samples)
nz = int(opt.nz)
nX = 2
sigma = 1
mu = 0

gauss_const = -np.log10(np.sqrt(2 * np.pi * (sigma**2)) ** nz)
log_const = np.log10(np.exp(1))
images = np.empty([samples, nX])
latent = np.empty([samples, nz])
prob = np.empty([samples])
jacob = np.empty([samples, nX])

# Generator
class mog_netG(nn.Module):
    def __init__(self, ngpu, nz, nX, dim):
        super(mog_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz,dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, nX),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# Encoder
class mog_netE(nn.Module):
    def __init__(self, ngpu, nz, nX, dim):
        super(mog_netE, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(2*nX, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, nz),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

if opt.gen:
    net = mog_netG(ngpu, nz, nX, opt.gendim)
    sign = -1
    epsilon = 5e-4
else:
    net = mog_netE(ngpu, nz, nX, opt.encdim)
    sign = 1
    epsilon = 1e-2
    temp = nX
    nX = nz
    nz = temp

filename = os.path.join(opt.outf, opt.netG)
if os.path.isfile(filename):
    print("==> loading params from checkpoint '{}'".format(filename))
    net.load_state_dict(torch.load(filename))
else:
    print("==> no checkpoint found at '{}'".format(filename))
    raise

print(net)

if not opt.gen:
    nz *= 2

a = torch.FloatTensor(1, nz)
b = torch.eye(nz) * epsilon

if opt.forward:
    noise = torch.FloatTensor(2*nz+1, nz)
else:
    noise = torch.FloatTensor(1, nz)

znoise = torch.FloatTensor(1, nz/2)

if opt.cuda:
    net.cuda()
    noise = noise.cuda()
    a = a.cuda()
    b = b.cuda()
    znoise = znoise.cuda()

net.eval()
dataset = generate_data_uniform()
for i in range(samples):
    J = np.empty([nX, nz])

    if opt.gen:
        a.normal_(0, 1)
    else:
        real_cpu, _ = dataset[i]
        real_cpu.resize_(1, nz/2)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        # znoise.normal_(0, 1)
        znoise.fill_(0)
        a.copy_(torch.cat((real_cpu, znoise), 1))

    # Generate sample noise
    if opt.forward:
        # Generate sequence of small perturbation in input noise variable z
        noise.copy_(torch.cat((a, a+b, a-b), 0))

        noisev = Variable(noise, volatile=True)
        fake = net(noisev)
        # if not opt.gen:
        #     nu, nsigma = fake[:, :nX], fake[:, nX:].exp()
        #     fake = nu + nsigma * znoise

        I = fake.data.cpu().numpy().reshape(2*nz+1,-1)

        J = (I[1:nz+1, :] - I[nz+1:, :]).transpose() / (2*epsilon)
    else:
        noise.copy_(a)
        noisev = Variable(noise, requires_grad=True)
        fake = net(noisev)
        fake = fake.view(1, -1)
        # if not opt.gen:
        #     nu, nsigma = fake[:, :nX], fake[:, nX:].exp()
        #     fake = nu + nsigma * znoise

        for k in range(nX):
            net.zero_grad()
            fake[0, k].backward(retain_variables=True)
            J[k] = noisev.grad.data.cpu().numpy().squeeze()
        I = fake.data.cpu().numpy()

    if opt.gen:
        images[i] = I[0, :]
        Z = a.cpu().numpy()
    else:
        images[i] = a.cpu().numpy()[:,:nz/2]
        Z = I[0, :]
    R = np.linalg.qr(J, mode='r')
    dummy = R.diagonal().copy()
    dummy[np.where(np.abs(dummy) < 1e-20)] = 1
    jacob[i] = dummy
    latent[i] = Z
    if opt.gen:
        prob[i] = -log_const * 0.5 * np.sum((Z-mu)**2) / (sigma**2) + gauss_const + sign * np.log10(np.abs(dummy)).sum()  # storing probabilities
    else:
        prob[i] = -log_const * 0.5 * (np.sum((Z - mu) ** 2)-np.sum((znoise.cpu().numpy() - mu) ** 2)) / (sigma ** 2) + sign * np.log10(
            np.abs(dummy)).sum()  # storing probabilities

print("The minimum value of prob is {} and the maximum is {}".format(min(prob), max(prob)))

sio.savemat(os.path.join(opt.outf, 'features'),
            {'feat': images.astype(np.float32),
             'Z': latent.astype(np.float32),
             'prob': prob.astype(np.float32),
             'jacob': jacob.astype(np.float32),
             })