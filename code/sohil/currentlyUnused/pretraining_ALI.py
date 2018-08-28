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
from itertools import chain
import numpy as np
import scipy.io as sio
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models import weights_init, generate_outlierexp_data, generate_classwise_data, AverageMeter

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--nef', type=int, default=32, help='hidden dimension of encoder')
parser.add_argument('--ngf', type=int, default=64, help='hidden dimension of generator')
parser.add_argument('--nzf', type=int, default=512, help='hidden dimension of discriminator for Z')
parser.add_argument('--ndf', type=int, default=1024, help='hidden dimension of discriminator')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.8')
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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
elif opt.dataset == 'mnist_outlier':
    dataset = generate_outlierexp_data(root=opt.dataroot,
                                       transform=transforms.Compose([transforms.ToPILImage(),transforms.Scale(opt.imageSize),]),
                                       transform2=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                                       size=opt.imageSize)
elif opt.dataset == 'mnist_classwise':
    dataset = generate_classwise_data(root=opt.dataroot, label=opt.classindex,
                                       transform=transforms.Compose([transforms.ToPILImage(),transforms.Scale(opt.imageSize),]),
                                       transform2=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                                       size=opt.imageSize)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nef = int(opt.nef)
nzf = int(opt.nzf)
nc = int(opt.nc)

# Generator
class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf*2) x 10 x 10
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf*2) x 13 x 13
            nn.ConvTranspose2d(ngf, ngf / 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf / 2),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf) x 28 x 28
            nn.ConvTranspose2d(ngf / 2, ngf / 2, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf / 2),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf / 2, ngf / 2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(ngf / 2),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf / 2, nc, 1, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# Encoder
class _netE(nn.Module):
    def __init__(self, ngpu, nz, nef, nc):
        super(_netE, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(nc, nef, 5, 1, 0, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf*4) x 4 x 4
            nn.Conv2d(nef, nef * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf*2) x 10 x 10
            nn.Conv2d(nef * 2, nef * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf*2) x 13 x 13
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf) x 28 x 28
            nn.Conv2d(nef * 8, nef * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(nef * 16, nef * 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(nef * 16),
            nn.LeakyReLU(negative_slope=0.1),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(nef * 16, 2 * nz, 1, 1, 0, bias=True),
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class convmaxout(nn.Module):
    def __init__(self, nchannels, numpieces):
        super(convmaxout, self).__init__()
        self.numfilters = nchannels / numpieces
        self.numpieces = numpieces

    def forward(self, input):
        siz = [input.size(0), self.numfilters, self.numpieces, input.size(2), input.size(3)]
        output, _ = input.view(siz).max(dim=2)
        return output

class maxout(nn.Module):
    def __init__(self, nchannels, numpieces):
        super(maxout, self).__init__()
        self.numfilters = nchannels / numpieces
        self.numpieces = numpieces

    def forward(self, input):
        siz = [input.size(0), self.numfilters, self.numpieces]
        output, _ = input.view(siz).max(dim=2)
        return output

# Discriminator
class _netD(nn.Module):
    def __init__(self, ngpu, nz, nef, ndf, nzf, nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.discX = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(nc, nef, 5, 1, 0, bias=False),
            nn.Dropout(p=0.2),
            convmaxout(nef, 2),
            # state size. (ngf*4) x 4 x 4
            nn.Conv2d(nef / 2, nef * 2, 4, 2, 0, bias=False),
            nn.Dropout(p=0.5),
            convmaxout(nef * 2, 2),
            # state size. (ngf*2) x 10 x 10
            nn.Conv2d(nef, nef * 4, 4, 1, 0, bias=False),
            nn.Dropout(p=0.5),
            convmaxout(nef * 4, 2),
            # state size. (ngf*2) x 13 x 13
            nn.Conv2d(nef * 2, nef * 8, 4, 2, 0, bias=False),
            nn.Dropout(p=0.5),
            convmaxout(nef * 8, 2),
            # state size. (ngf) x 28 x 28
            nn.Conv2d(nef * 4, nef * 16, 4, 1, 0, bias=False),
            nn.Dropout(p=0.5),
            convmaxout(nef * 16, 2),
        )
        self.discZ = nn.Sequential(
            nn.Linear(nz, nzf, bias=False),
            nn.Dropout(p=0.2),
            maxout(nzf, 2),
            nn.Linear(nzf/2, nzf, bias=False),
            nn.Dropout(p=0.5),
            maxout(nzf, 2),
        )
        self.joint = nn.Sequential(
            nn.Linear(nef * 8 + (nzf/2), ndf, bias=False),
            nn.Dropout(p=0.5),
            maxout(ndf, 2),
            nn.Linear(ndf / 2, ndf, bias=False),
            nn.Dropout(p=0.5),
            maxout(ndf, 2),
            nn.Linear(ndf / 2, 1, bias=False),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )

    def forward(self, image, latent):
        latent = latent.view(latent.size(0), latent.size(1))
        if isinstance(image.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputX = nn.parallel.data_parallel(self.discX, image, range(self.ngpu))
            outputZ = nn.parallel.data_parallel(self.discZ, latent, range(self.ngpu))
            output = nn.parallel.data_parallel(self.joint, torch.cat([outputX.view(outputX.size(0), -1), outputZ],
                                                                     dim=1), range(self.ngpu))
        else:
            outputX = self.discX(image)
            outputZ = self.discZ(latent)
            output = self.joint(torch.cat([outputX.view(outputX.size(0), -1), outputZ], dim=1))

        return output.view(-1, 1).squeeze(1)

netG = _netG(ngpu, nz, ngf, nc)
netG.apply(weights_init)

netE = _netE(ngpu, nz, nef, nc)
netE.apply(weights_init)

netD = _netD(ngpu, nz, nef, ndf, nzf, nc)
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

x = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
z = torch.FloatTensor(opt.batchSize, nz, 1, 1)
noise = torch.FloatTensor(opt.batchSize, nz)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netE.cuda()
    x, z, noise = x.cuda(), z.cuda(), noise.cuda()

x, z, noise = Variable(x), Variable(z), Variable(noise)

def sampling():
    z_hat = netE(x)
    mu, sigma = z_hat[:, :nz, 0, 0], z_hat[:, nz:, 0, 0].exp()
    z_hat = mu + sigma * noise
    z_hat = z_hat.view(x.size(0), nz, 1, 1)

    x_hat = netG(z)

    x_recon = netG(z_hat)

    return x_hat, z_hat, x_recon

def compute_loss():

    x_hat, z_hat, _ = sampling()

    data_preds = netD(x, z_hat)
    sample_preds = netD(x_hat, z)

    # generator loss
    gloss = torch.mean(F.softplus(data_preds) + F.softplus(-sample_preds))

    data_preds = netD(x, z_hat.detach())
    sample_preds = netD(x_hat.detach(), z)

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
    dis_loss = AverageMeter()
    gen_loss = AverageMeter()
    real_out = AverageMeter()
    fake_out = AverageMeter()

    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data

        netD.zero_grad()
        netG.zero_grad()
        netE.zero_grad()

        batch_size = real_cpu.size(0)

        if opt.cuda:
            real_cpu = real_cpu.cuda()

        x.data.resize_as_(real_cpu).copy_(real_cpu)
        z.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noise.data.resize_(batch_size, nz).normal_(0, 1)

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

        dis_loss.update(D_loss.data[0], batch_size)
        gen_loss.update(G_loss.data[0], batch_size)
        real_out.update(Dx, batch_size)
        fake_out.update(D_G_z1, batch_size)

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
          % (epoch, opt.niter, dis_loss.avg, gen_loss.avg, real_out.avg, fake_out.avg))

    if opt.tensorboard:
        log_value('Discriminator_loss', dis_loss.avg, epoch)
        log_value('GenEnc_loss', gen_loss.avg, epoch)
        log_value('Real_output', real_out.avg, epoch)
        log_value('Fake_output', fake_out.avg, epoch)

# do checkpointing
torch.save(netG.state_dict(), '%s/netG_ALI_%s.pth' % (opt.outf, opt.dataset))
torch.save(netD.state_dict(), '%s/netD_ALI_%s.pth' % (opt.outf, opt.dataset))
torch.save(netE.state_dict(), '%s/netE_ALI_%s.pth' % (opt.outf, opt.dataset))

#### Testing ####
netD.eval()
netG.eval()
netE.eval()

x_real = np.empty([len(dataset), nc, opt.imageSize, opt.imageSize])
x_sampled = np.empty([len(dataset), nc, opt.imageSize, opt.imageSize])
x_recon = np.empty([len(dataset), nc, opt.imageSize, opt.imageSize])

for i, (real_cpu, labels) in enumerate(dataloader, 0):
    batch_size = real_cpu.size(0)
    x_real[i*opt.batchSize:(i+1)*opt.batchSize] = real_cpu.numpy()

    if opt.cuda:
        real_cpu = real_cpu.cuda()

    x.data.resize_as_(real_cpu).copy_(real_cpu)
    z.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
    noise.data.resize_(batch_size, nz).normal_(0, 1)

    x_hat, z_hat, x_tilde = sampling()
    x_sampled[i * opt.batchSize:(i + 1) * opt.batchSize] = x_hat.data.cpu().numpy()
    x_recon[i * opt.batchSize:(i + 1) * opt.batchSize] = x_tilde.data.cpu().numpy()

sio.savemat(os.path.join(opt.outf, 'ALI_%s.mat' % (opt.dataset)),
            {'real':x_real.astype(np.float32),
            'fake':x_sampled.astype(np.float32),
             'recon':x_recon.astype(np.float32),
             })