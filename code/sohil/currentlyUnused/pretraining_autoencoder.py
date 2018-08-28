from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import Q_net, P_net, D_net_gauss, weights_init, generate_data, mog_netD, mog_netG, AverageMeter

from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=1000)
parser.add_argument('--ndf', type=int, default=1000)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--gen_lr', type=float, default=0.01, help='learning rate, default=0.0006')
parser.add_argument('--reg_lr', type=float, default=0.1, help='learning rate, default=0.0008')
parser.add_argument('--sigma', type=float, default=5, help='variance for gaussian. default=5')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netQ', default='', help="path to netQ (to continue training)")
parser.add_argument('--netP', default='', help="path to netP (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number')

opt = parser.parse_args()
print(opt)

if opt.tensorboard:
    configure("/fs/vulcan-scratch/sohil/distGAN/runs/pretraining/%s" % (opt.id))

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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nX = opt.imageSize * opt.imageSize * 3
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nX = opt.imageSize * opt.imageSize * 3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nX = opt.imageSize * opt.imageSize * 3
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nX = opt.imageSize * opt.imageSize
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nX = opt.imageSize * opt.imageSize * 3
elif opt.dataset == 'mog':
    dataset = generate_data(num_mode=8,except_num=2)
    nX = 2

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

if opt.dataset == 'mog':
    netQ = mog_netG(ngpu, nz)
    netD = mog_netD(ngpu)
else:
    netQ = Q_net(nX,ngf,nz)
    netP = P_net(nX,ndf,nz)
    netD = D_net_gauss(nz,ndf)

netQ.apply(weights_init)
netP.apply(weights_init)
netD.apply(weights_init)

if opt.netQ != '':
    netQ.load_state_dict(torch.load(opt.netQ))
print(netQ)

if opt.netP != '':
    netP.load_state_dict(torch.load(opt.netP))
print(netP)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, nX)
noise = torch.FloatTensor(opt.batchSize, nz)

label_rl = torch.FloatTensor(opt.batchSize, nX)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

criterion1 = nn.BCELoss()
criterion2 = nn.MSELoss(size_average=False)
if opt.cuda:
    netP.cuda()
    netQ.cuda()
    netD.cuda()
    criterion1.cuda()
    criterion2.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()
    label_rl = label_rl.cuda()

gen_lr, reg_lr = opt.gen_lr, opt.reg_lr

# Set optimizators
P_decoder = optim.SGD(netP.parameters(), lr=gen_lr, momentum=opt.beta1, nesterov=True)
Q_encoder = optim.SGD(netQ.parameters(), lr=gen_lr, momentum=opt.beta1, nesterov=True)
Q_generator = optim.SGD(netQ.parameters(), lr=reg_lr, momentum=0)
D_gauss_solver = optim.SGD(netD.parameters(), lr=reg_lr, momentum=0)
P_scheduler = lr_scheduler.MultiStepLR(P_decoder, milestones=[50], gamma=0.1)
Q_scheduler = lr_scheduler.MultiStepLR(Q_encoder, milestones=[50], gamma=0.1)

for epoch in range(opt.niter):
    err_recon = AverageMeter()
    err_dis = AverageMeter()
    err_gen = AverageMeter()
    val_D_x = AverageMeter()
    val_D_G_z1 = AverageMeter()
    val_D_G_z2 = AverageMeter()
    P_scheduler.step()
    Q_scheduler.step()


    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update P,Q network: reconstruction loss
        ###########################
        netQ.zero_grad()
        netP.zero_grad()

        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        real_cpu = real_cpu.view(batch_size, -1)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label_rl.resize_as_(real_cpu).copy_(real_cpu)

        inputv = Variable(input)
        labelv_rl = Variable(label_rl)

        z_sample = netQ(inputv)
        X_sample = netP(z_sample)
        recon_loss = criterion2(X_sample, labelv_rl) / real_cpu.size(0)

        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()

        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()

        z_real_gauss = torch.randn(batch_size, nz) * opt.sigma
        if opt.cuda:
            z_real_gauss = z_real_gauss.cuda()
        noise.resize_as_(z_real_gauss).copy_(z_real_gauss)

        label.resize_(batch_size).fill_(real_label)
        noisev = Variable(noise)
        labelv = Variable(label)

        output = netD(noisev)
        errD_real = criterion1(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        netQ.eval()
        fake_noise = netQ(inputv)

        labelv = Variable(label.fill_(fake_label))
        output = netD(fake_noise)
        errD_fake = criterion1(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        D_gauss_solver.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        netQ.train()
        netQ.zero_grad()
        fake_noise = netQ(inputv)

        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake_noise)
        errG = criterion1(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        Q_generator.step()

        err_recon.update(recon_loss.data[0]/batch_size, batch_size)
        err_dis.update(errD.data[0], batch_size)
        err_gen.update(errG.data[0], batch_size)
        val_D_x.update(D_x, batch_size)
        val_D_G_z1.update(D_G_z1, batch_size)
        val_D_G_z2.update(D_G_z2, batch_size)

    if opt.tensorboard:
        log_value('reconstruction error', err_recon.avg, epoch)
        log_value('discriminator error', err_dis.avg, epoch)
        log_value('generator error', err_gen.avg, epoch)
        log_value('D_x', val_D_x.avg, epoch)
        log_value('D_G_z1', val_D_G_z1.avg, epoch)
        log_value('D_G_z2', val_D_G_z2.avg, epoch)

# do checkpointing
torch.save(netP.state_dict(), '%s/AE_netP_epoch_%d.pth' % (opt.outf, epoch))
torch.save(netQ.state_dict(), '%s/AE_netQ_epoch_%d.pth' % (opt.outf, epoch))
torch.save(netD.state_dict(), '%s/AE_netD_epoch_%d.pth' % (opt.outf, epoch))