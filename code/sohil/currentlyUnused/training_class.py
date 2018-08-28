from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from models import _lenet, init_params, _lenet_ad, generate_outlierexp_data
import torchvision.models as modelss

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netP', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/fs/vulcan-scratch/sohil/distGAN/checkpoints',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number')

def main():
    global opt
    opt = parser.parse_args()
    print(opt)

    if opt.tensorboard:
        configure("/fs/vulcan-scratch/sohil/distGAN/runs/%s" % (opt.id))

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
        trainset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ]))
        testset = dset.CIFAR10(root=opt.dataroot, train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ]))
    elif opt.dataset == 'mnist':
        mnist_trans = transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = dset.MNIST(root=opt.dataroot, download=True, transform=mnist_trans)
        testset = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=mnist_trans)
    elif opt.dataset == 'mnist_outlier':
        trainset = generate_outlierexp_data(root=opt.dataroot,
                                           transform=transforms.Compose(
                                               [transforms.ToPILImage(), transforms.Scale(opt.imageSize), ]),
                                           transform2=transforms.Compose(
                                               [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                           size=opt.imageSize, train=True)
        testset = generate_outlierexp_data(root=opt.dataroot,
                                           transform=transforms.Compose(
                                               [transforms.ToPILImage(), transforms.Scale(opt.imageSize), ]),
                                           transform2=transforms.Compose(
                                               [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                           size=opt.imageSize, train=False)
    assert trainset
    assert testset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    ngpu = int(opt.ngpu)

    if opt.dataset == 'mnist' or opt.dataset == 'mnist_outlier':
        netP = _lenet_ad(ngpu)
        init_params(netP)
    elif opt.dataset == 'cifar10':
        netP = modelss.vgg16_bn(num_classes=10)

    if opt.netP != '':
        netP.load_state_dict(torch.load(opt.netD))
    print(netP)

    criterion = nn.CrossEntropyLoss(size_average=True)

    if opt.cuda:
        netP.cuda()
        criterion.cuda()

    # setup optimizer
    optimizerP = optim.Adam(netP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizerP, step_size=50, gamma=0.5)

    for epoch in range(opt.niter):
        scheduler.step()
        train(trainloader, netP, optimizerP, criterion, opt.cuda, epoch)
        test(testloader, netP, criterion, opt.cuda, epoch)

        # do checkpointing
        torch.save(netP.state_dict(), '%s/%s_disc_epoch_%d.pth' % (opt.outf, opt.dataset, epoch))

def train(dataloader, net, optimizer, criterion, use_cuda, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()

    net.train()

    for i, (data, label) in enumerate(dataloader):
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        datav = Variable(data)
        labelv = Variable(label)

        _, output = net(datav)
        err = criterion(output, labelv)

        losses.update(err.data[0], data.size(0))
        _, predicted = torch.max(output.data, 1)
        accuracy.update(predicted.eq(label).float().mean(), data.size(0))

        err.backward()
        optimizer.step()

    if opt.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', accuracy.avg, epoch)


def test(dataloader, net, criterion, use_cuda, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()

    net.eval()

    for i, (data, label) in enumerate(dataloader):
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        datav = Variable(data, volatile=True)
        labelv = Variable(label, volatile=True)

        _,output = net(datav)
        err = criterion(output, labelv)

        losses.update(err.data[0], data.size(0))
        _, predicted = torch.max(output.data, 1)
        accuracy.update(predicted.eq(label).float().mean(), data.size(0))

    if opt.tensorboard:
        log_value('test_loss', losses.avg, epoch)
        log_value('test_acc', accuracy.avg, epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()