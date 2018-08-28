from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from models import weights_init, generate_data_single_batch, mog_netD, mog_netG


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--nz', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

netG = mog_netG(ngpu, nz)
netG.apply(weights_init)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = mog_netD(ngpu)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 2)
noise = torch.FloatTensor(opt.batchSize, nz)
fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_(0, 1)

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    real_cpu = generate_data_single_batch(num_mode=8, except_num=2)

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real

    netD.zero_grad()
    batch_size = real_cpu.size(0)
    if opt.cuda:
        real_cpu = real_cpu.cuda()
    input.resize_as_(real_cpu).copy_(real_cpu)
    label.resize_(batch_size).fill_(real_label)
    inputv = Variable(input)
    labelv = Variable(label)

    output = netD(inputv)
    errD_real = criterion(output, labelv)
    errD_real.backward()
    D_x = output.data.mean()

    # train with fake
    noise.resize_(batch_size, nz).normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev)
    labelv = Variable(label.fill_(fake_label))
    output = netD(fake.detach()) # this just saves computation time
    errD_fake = criterion(output, labelv)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
    output = netD(fake)
    errG = criterion(output, labelv)
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          % (epoch, opt.niter, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

# do checkpointing
torch.save(netG.state_dict(), '%s/netG_epoch_mog.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD_epoch_mog.pth' % (opt.outf))