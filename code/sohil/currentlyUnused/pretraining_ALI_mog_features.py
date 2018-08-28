from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from models import weights_init, generate_data_single_batch, generate_data
from itertools import chain
import numpy as np
import scipy.io as sio

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--nz', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--encdim', type=int, default=200, help='hidden dimension of encoder')
parser.add_argument('--gendim', type=int, default=200, help='hidden dimension of generator')
parser.add_argument('--discdim', type=int, default=400, help='hidden dimension of discriminator')
parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.8')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number')

opt = parser.parse_args()
print(opt)

if opt.tensorboard:
    configure("/fs/vulcan-scratch/sohil/distGAN/runs/%s" % (opt.id))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if opt.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
nX = 2

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

# Discriminator
class mog_netD(nn.Module):
    def __init__(self, ngpu, indim, hidim):
        super(mog_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(indim, hidim),
            nn.Tanh(),
            nn.Linear(hidim, hidim),
            nn.Tanh(),
            nn.Linear(hidim, hidim),
            nn.Tanh(),
        )
        self.final = nn.Sequential(
            nn.Linear(hidim, hidim),
            nn.Tanh(),
            nn.Linear(hidim, 1),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            inter = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.final, inter, range(self.ngpu))
        else:
            inter = self.main(input)
            output = self.final(inter)

        return inter, output.view(-1, 1)

netG = mog_netG(ngpu, nz, nX, opt.gendim)
netG.apply(weights_init)

netE = mog_netE(ngpu, nz, nX, opt.encdim)
netE.apply(weights_init)

netD = mog_netD(ngpu, nz+nX, opt.discdim)
netD.apply(weights_init)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

x = torch.FloatTensor(opt.batchSize, nX)
z = torch.FloatTensor(opt.batchSize, nz)
noise = torch.FloatTensor(opt.batchSize, nz)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netE.cuda()
    x, z, noise = x.cuda(), z.cuda(), noise.cuda()

x, z, noise = Variable(x), Variable(z), Variable(noise)

def sampling():
    # z_hat = netE(x)
    # mu, sigma = z_hat[:, :opt.nz], z_hat[:, opt.nz:].exp()
    # z_hat = mu + sigma * noise

    z_hat = netE(torch.cat([x, noise],1))

    x_hat = netG(z)

    x_recon = netG(z_hat)

    return x_hat, z_hat, x_recon

def compute_loss():

    x_hat, z_hat, _ = sampling()

    data_preds, _ = netD(torch.cat([x, z_hat], 1))
    sample_preds, _ = netD(torch.cat([x_hat, z], 1))

    # generator loss
    # gloss = torch.norm(torch.mean(data_preds, 0) - torch.mean(sample_preds, 0), p=2)**2
    gloss = torch.mean(torch.abs(torch.mean(data_preds, 0) - torch.mean(sample_preds, 0)))

    _, data_preds = netD(torch.cat([x, z_hat.detach()], 1))
    _, sample_preds = netD(torch.cat([x_hat.detach(), z], 1))

    # discriminator loss
    dloss = torch.mean(F.softplus(-data_preds) + F.softplus(sample_preds))

    return gloss, dloss, data_preds, sample_preds

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(chain(netG.parameters(), netE.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

netD.train()
netG.train()
netE.train()

for epoch in range(opt.niter):
    real_cpu = generate_data_single_batch(num_mode=8, except_num=1, radius=2, batch_size=opt.batchSize)

    netD.zero_grad()
    netG.zero_grad()
    netE.zero_grad()

    batch_size = real_cpu.size(0)

    if opt.cuda:
        real_cpu = real_cpu.cuda()

    x.data.resize_as_(real_cpu).copy_(real_cpu)
    z.data.resize_(batch_size, nz).normal_(0,1)
    noise.data.resize_(batch_size, nz).normal_(0,1)

    # Simultaneous updates
    G_loss, D_loss, Dx, D_G_z1 = compute_loss()

    Dx = Dx.data.mean()
    D_G_z1 = D_G_z1.data.mean()

    # First compute gradient w.r.t. Generator loss
    G_loss.backward()

    # Then clear gradients from Discriminator network and compute gradients
    # This works because generator and encoder n/w's are detached.
    netD.zero_grad()
    D_loss.backward()

    # Finally simultaneously update both networks
    optimizerD.step()  # Apply optimization step
    optimizerG.step()  # Apply optimization step

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
          % (epoch, opt.niter, D_loss.data[0], G_loss.data[0], Dx, D_G_z1))

    if opt.tensorboard:
        log_value('Discriminator_loss', D_loss.data[0], epoch)
        log_value('GenEnc_loss', G_loss.data[0], epoch)
        log_value('Real_output', Dx, epoch)
        log_value('Fake_output', D_G_z1, epoch)

# do checkpointing
torch.save(netG.state_dict(), '%s/netG_ALI_mog_features.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD_ALI_mog_features.pth' % (opt.outf))
torch.save(netE.state_dict(), '%s/netE_ALI_mog_features.pth' % (opt.outf))

#### Testing ####
netD.eval()
netG.eval()
netE.eval()

dataset = generate_data(num_mode=8, except_num=1, radius=2, num_data_per_class=10000)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=0)

x_real = np.empty([len(dataset), nX])
x_sampled = np.empty([len(dataset), nX])
x_recon = np.empty([len(dataset), nX])

for i, (real_cpu,labels) in enumerate(dataloader, 0):
    batch_size = real_cpu.size(0)
    x_real[i*opt.batchSize:(i+1)*opt.batchSize] = real_cpu.numpy().reshape(batch_size,nX)

    if opt.cuda:
        real_cpu = real_cpu.cuda()

    x.data.resize_as_(real_cpu).copy_(real_cpu)
    z.data.resize_(batch_size, nz).normal_(0,1)
    noise.data.resize_(batch_size, nz).normal_(0,1)

    x_hat, z_hat, x_tilde = sampling()
    x_sampled[i * opt.batchSize:(i + 1) * opt.batchSize] = x_hat.data.cpu().numpy().reshape(batch_size, nX)
    x_recon[i * opt.batchSize:(i + 1) * opt.batchSize] = x_tilde.data.cpu().numpy().reshape(batch_size, nX)

sio.savemat(os.path.join(opt.outf, 'ALI_MOG.mat'),
            {'real':x_real.astype(np.float32),
            'fake':x_sampled.astype(np.float32),
             'recon':x_recon.astype(np.float32),
             })