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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import _netP, generate_data, mog_netP, generate_adversarial_data, generate_data_uniform


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netP', default='', help="path to netP (to load model)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

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

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mog':
    # dataset = generate_data(num_mode=8,except_num=1,num_data_per_class=10000)
    dataset = generate_data_uniform()

elif opt.dataset == 'adversarial':
    dataset = generate_adversarial_data(root=opt.dataroot, transform=transforms.Compose([transforms.ToPILImage(),
                               transforms.Scale(opt.imageSize),]), transform2=transforms.Compose([
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
ndf = int(opt.ndf)
nc = int(opt.nc)

if opt.dataset == 'mog':
    netP = mog_netP(ngpu)
else:
    netP = _netP(ngpu, nc, ndf)

filename = os.path.join(opt.outf, opt.netP)
if os.path.isfile(filename):
    print("==> loading params from checkpoint '{}'".format(filename))
    netP.load_state_dict(torch.load(filename))
else:
    print("==> no checkpoint found at '{}'".format(filename))
    raise

print(netP)

if opt.cuda:
    netP.cuda()

samples = len(dataset)
if opt.dataset == 'mog':
    images = np.empty([samples,2])
else:
    images = np.empty([samples,nc,opt.imageSize,opt.imageSize])
prob = np.empty([samples])
label = np.empty([samples])

netP.eval()
for i, (data,labels) in enumerate(dataloader, 0):
    # # added random noise for adversarial exmaples
    # data = torch.add(data,0.1,torch.randn(data.shape))

    if opt.cuda:
        datav = data.cuda()

    datav = Variable(datav, volatile=True)
    output = netP(datav, 5)

    prob[i*opt.batchSize:(i+1)*opt.batchSize] = output.data.cpu().numpy().reshape(data.size(0))
    # label[i * opt.batchSize:(i + 1) * opt.batchSize] = labels.numpy().reshape(data.size(0))
    if opt.dataset == 'mog':
        images[i*opt.batchSize:(i+1)*opt.batchSize] =  data.numpy().reshape(data.size(0),2)
    else:
        images[i*opt.batchSize:(i+1)*opt.batchSize] = data.numpy().reshape(data.size(0),nc,opt.imageSize,opt.imageSize) # storing image

print("The minimum value of prob is {} and the maximum is {}".format(min(prob), max(prob)))

sio.savemat(os.path.join(opt.outf, 'testoutput.mat'),
            {'images':images.astype(np.float32),
            'prob':prob.astype(np.float32),
             })

# this section is for saving best and worst images
if opt.dataset != 'mog':
    indices = np.argsort(prob)
    vutils.save_image(torch.from_numpy(images[indices[:-101:-1]]), os.path.join(opt.outf, 'best_'+opt.dataset+'_image.png'),
                      nrow=10, normalize=True)
    vutils.save_image(torch.from_numpy(images[indices[:100]]), os.path.join(opt.outf, 'worst_'+opt.dataset+'_image.png'),
                      nrow=10, normalize=True)