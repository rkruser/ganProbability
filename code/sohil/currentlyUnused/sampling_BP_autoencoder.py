from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import scipy.io as sio
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import Q_net


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=5, help='variance for gaussian. default=5')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netQ', default='', help="path to netQ (to load model)")
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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nX = 256*256*3
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_test'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nX = 64 * 64 * 3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nX = 32 * 32 * 3
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nX = 28 * 28

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)

netQ = Q_net(nX,ngf,nz)

filename = os.path.join(opt.outf, opt.netQ)
if os.path.isfile(filename):
    print("==> loading params from checkpoint '{}'".format(filename))
    netQ.load_state_dict(torch.load(filename))
else:
    print("==> no checkpoint found at '{}'".format(filename))
    raise

print(netQ)

epsilon = 2e-1
b = torch.eye(nX) * epsilon
input = torch.FloatTensor(2*nX + 1, nX)

if opt.cuda:
    netQ.cuda()
    b = b.cuda()
    input = input.cuda()

samples = len(dataset)
gauss_const = -np.log10(np.sqrt(2*np.pi*(opt.sigma**2))**nz)
log_const = np.log10(np.exp(1))
images = np.empty([samples,nX])
labels = np.empty([samples])
prob = np.empty([samples])
jacob = np.empty([samples,nz])

netQ.eval()
for i, data in enumerate(dataloader, 0):
    J = np.empty([nz,nX])
    real_cpu,label = data
    labels[i] = label.numpy()
    real_cpu = real_cpu.view(1,nX)
    images[i] = real_cpu.squeeze(0).numpy()
    if opt.cuda:
        real_cpu = real_cpu.cuda()

    if opt.forward:
        # Generate sequence of small perturbation in input noise variable z
        input.copy_(torch.cat((real_cpu, real_cpu+b, real_cpu-b),0))

        inputv = Variable(input, volatile=True)
        a = netQ(inputv)

        I = a.data.cpu().numpy().reshape(2*nX+1,-1)
        Z = I[0,:]

        J = (I[1:nX+1,:] - I[nX+1:, :]).transpose() / (2*epsilon)
    else:
        inputv = Variable(real_cpu, requires_grad=True)
        a = netQ(inputv)
        for k in range(nz):
            netQ.zero_grad()
            a[0,k].backward(retain_variables=True)
            J[k] = inputv.grad.data.cpu().numpy().squeeze(0)
        Z = a.data.cpu().numpy().squeeze(0)

    R = np.linalg.qr(J, mode='r')
    dummy = R.diagonal().copy()
    # dummy[np.where(np.abs(dummy) < 1e-20)] = 1
    jacob[i] = dummy
    prob[i] = -log_const * 0.5 * np.sum(Z**2) / opt.sigma**2 + gauss_const + np.log10(np.abs(dummy)).sum()  # storing probabilities

print("The minimum value of prob is {} and the maximum is {}".format(min(prob), max(prob)))

sio.savemat(os.path.join(opt.outf, 'trainoutput.mat'),
            {'images':images.astype(np.float32),
             'labels':labels,
            'jacob': jacob.astype(np.float32),
            'prob':prob.astype(np.float32),
             })

indices = np.arange(len(prob))

vutils.save_image(torch.from_numpy(images[indices[:-101:-1]]),
                  os.path.join(opt.outf, 'best_{}_test_image.png'.format(opt.dataset)),nrow=10,normalize=True)
vutils.save_image(torch.from_numpy(images[indices[:100]]),
                  os.path.join(opt.outf, 'worst_{}_test_image.png'.format(opt.dataset)),nrow=10,normalize=True)