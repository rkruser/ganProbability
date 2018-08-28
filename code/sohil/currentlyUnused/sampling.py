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
import torchvision.utils as vutils
from torch.autograd import Variable
from models import _netG


parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--samples', type=int, default=10000)
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--classindex', type=int, default=0)
parser.add_argument('--netG', default='', help="path to netG (to load model)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--forward', help='Computes Jacobian using Forward pass', action='store_true')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = int(opt.nc)

netG = _netG(ngpu, nz,ngf,nc)

filename = os.path.join(opt.outf, opt.netG)
if os.path.isfile(filename):
    print("==> loading params from checkpoint '{}'".format(filename))
    netG.load_state_dict(torch.load(filename))
else:
    print("==> no checkpoint found at '{}'".format(filename))
    raise

print(netG)

epsilon = 1e-5
if opt.forward:
    noise = torch.FloatTensor(2*nz+1, nz, 1, 1)
else:
    noise = torch.FloatTensor(1, nz, 1, 1)
a = torch.FloatTensor(1,nz)
b = torch.eye(nz) * epsilon

if opt.cuda:
    netG.cuda()
    noise = noise.cuda()
    a = a.cuda()
    b = b.cuda()

samples = int(opt.samples)
gauss_const = -np.log10(np.sqrt(2*np.pi)**nz)
log_const = np.log10(np.exp(1))
images = np.empty([samples,nc,opt.imageSize,opt.imageSize])
prob = np.empty([samples])
jacob = np.empty([samples,nz])
code = np.empty([samples,nz])
nX = nc*opt.imageSize**2

netG.eval()
for i in range(samples):
    J = np.empty([nX, nz])
    # Generate sample noise
    a.normal_(0,1)
    if opt.forward:
        # Generate sequence of small perturbation in input noise variable z
        noise.copy_(torch.cat((a, a+b, a-b),0).unsqueeze(2).unsqueeze(3))

        noisev = Variable(noise, volatile=True)
        fake = netG(noisev)

        I = fake.data.cpu().numpy().reshape(2*nz+1,-1)

        J = (I[1:nz+1,:] - I[nz+1:, :]).transpose() / (2*epsilon)
    else:
        noise.copy_(a.unsqueeze(2).unsqueeze(3))

        noisev = Variable(noise, requires_grad=True)
        fake = netG(noisev)
        fake = fake.view(1,-1)

        for k in range(nX):
            netG.zero_grad()
            fake[0,k].backward(retain_variables=True)
            J[k] = noisev.grad.data.cpu().numpy().squeeze()
        I = fake.data.cpu().numpy()

    images[i] = I[0,:].reshape(nc,opt.imageSize,opt.imageSize) # storing image
    R = np.linalg.qr(J, mode='r')
    Z = a.cpu().numpy()
    code[i] = Z.squeeze()
    dummy = R.diagonal().copy()
    dummy[np.where(np.abs(dummy) < 1e-20)] = 1
    jacob[i] = dummy
    prob[i] = -log_const * 0.5 * np.sum(Z**2) + gauss_const - np.log10(np.abs(dummy)).sum()  # storing probabilities

print("The minimum value of prob is {} and the maximum is {}".format(min(prob), max(prob)))

sio.savemat(os.path.join(opt.outf, 'features_%d.mat' % opt.classindex),
            {'images': images.astype(np.float32),
             'code': code.astype(np.float32),
             'prob': prob.astype(np.float32),
             'jacob': jacob.astype(np.float32),
             })

# this section is for saving best and worst images
# data = np.load('ProbDistGAN/features.npz')
# prob = data['prob']
# images = data['images']
# images = np.squeeze(images,1)
# prob = np.squeeze(prob,0)
# # indices = np.argsort(prob)
# indices = np.arange(len(prob))
# vutils.save_image(torch.from_numpy(images[indices[:-101:-1]]),'ProbDistGAN/best_fake_image.png',nrow=10,normalize=True)
# vutils.save_image(torch.from_numpy(images[indices[:100]]),'ProbDistGAN/worst_fake_image.png',nrow=10,normalize=True)