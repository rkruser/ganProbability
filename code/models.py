import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.utils.data import TensorDataset
from PIL import Image
import torchvision.transforms as transforms

# Every model for DCGAN - Generator
class _netG(nn.Module):
    def __init__(self, ngpu, nz,ngf,nc):
        super(_netG, self).__init__()
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
            # state size. (ngf*2) x 8 x 8
            # nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            # for 28 x 28
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
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

# Discriminator
class _netD(nn.Module):
    def __init__(self, ngpu,nc,ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
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
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# Regressor
class _netP(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netP, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            # for 28 x 28
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=True),
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


# New nonlinear layer
class tomg(nn.Module):
    def __init__(self, dim):
        super(tomg, self).__init__()
        self.b = nn.Parameter(torch.Tensor(dim))
        # Initialization
        self.b.data = torch.zeros(dim)

    def forward(self, input):
        output = -torch.abs(input) + self.b
        return output

class sohil(nn.Module):
    def __init__(self, dim):
        super(sohil, self).__init__()
        self.b = nn.Parameter(torch.Tensor(dim))
        self.scale = nn.Parameter(torch.Tensor(dim))
        # Initialization
        self.b.data = torch.zeros(dim)
        self.scale.data = torch.ones(dim)

    def forward(self, input):
        output = input - self.scale * F.relu(input - self.b, inplace=True)
        # output = input - F.relu(input - self.b, inplace=True)
        return output

# Regressor
class mog_netP(nn.Module):
    def __init__(self, ngpu):
        super(mog_netP, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            # tomg(256),
            nn.Linear(256, 1),
            sohil(1),
        )

    def forward(self, input, th):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# Every model for Autoencoder - Encoder
class Q_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = self.lin1(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

# Decoder
class P_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, z_dim, N):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = self.lin1(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

# Sampling data
class prob_data(data.Dataset):
    """Custom Dataset loader for Probability training"""

    def __init__(self, root, name=None, label=None, train=True):
        self.root_dir = root
        self.train = train
        self.ntrainsamples = 40000

        if name is not None:
          data = sio.loadmat(osp.join(root,name))
        else:
          if label is not None:
              data = sio.loadmat(osp.join(root, 'features_%d.mat' % label))
          else:
              data = sio.loadmat(osp.join(root, 'features.mat'))
        # data = np.load(osp.join(root, 'features.npz'))
        if self.train:
            # self.train_data = data['images'][:self.ntrainsamples].astype(np.float32)
            self.train_data = data['images'][:self.ntrainsamples].astype(np.float32)
            self.train_prob = np.squeeze(data['prob'][0,:self.ntrainsamples].astype(np.float32))
        else:
            # self.test_data = data['images'][self.ntrainsamples:].astype(np.float32)
            self.test_data = data['images'][self.ntrainsamples:].astype(np.float32)
            self.test_prob = np.squeeze(data['prob'][0,self.ntrainsamples:].astype(np.float32))

    def __len__(self):
        if self.train:
            return self.ntrainsamples
        else:
            return len(self.test_prob)

    def __getitem__(self, item):
        if self.train:
            data, target = self.train_data[item], self.train_prob[item]
        else:
            data, target = self.test_data[item], self.test_prob[item]

        return data, target

# Sampling adversarial data
class generate_adversarial_data(data.Dataset):
    """Custom Dataset loader for adversarial training"""

    def __init__(self, root, transform=None, transform2=None, size=32):
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        data = sio.loadmat(osp.join(root, 'mnist_adversarial.mat'))
        self.data = data['data'].astype(np.float32).transpose(0,2,3,1)
        self.size=size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = np.zeros(len(data))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # data = transforms.ToPILImage(data)
        # data = Image.fromarray(data, mode='F')

        if self.transform is not None:
            data = self.transform(data)

        data = torch.from_numpy(np.array(data, copy = False))
        data = data.view(self.size,self.size,1)
        data = data.transpose(0, 1).transpose(0, 2).contiguous()

        if self.transform2 is not None:
            data = self.transform2(data)

        return data, target

# Sampler for classwise training of GAN
class generate_classwise_data(data.Dataset):
    """Custom Dataset loader for outlier exp"""

    def __init__(self, root, label, transform=None, transform2=None, size=28, train=True):
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        if train:
            data = sio.loadmat(osp.join(root, 'mnist.mat'))
        else:
            data = sio.loadmat(osp.join(root, 'test.mat'))
        self.data = data['X'].astype(np.float32)#.reshape((-1,28,28,1))
        self.label = data['Y'].astype(int).squeeze()
        self.data = self.data[self.label == label]
        self.label = self.label[self.label == label]
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.label[item]

        if self.transform is not None:
            data = self.transform(data)

        data = torch.from_numpy(data)
#        data = torch.from_numpy(np.array(data, copy=False))
#        data = data.view(self.size, self.size, 1)
#        data = data.transpose(0, 1).transpose(0, 2).contiguous()

        if self.transform2 is not None:
            data = self.transform2(data)

        return data, target

# Sampling mnist data for outlier experiment
class generate_outlierexp_data(data.Dataset):
    """Custom Dataset loader for outlier exp"""

    def __init__(self, root, transform=None, transform2=None, size=28, train=True):
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        if train:
            data = sio.loadmat(osp.join(root, 'mnist.mat'))
        else:
            data = sio.loadmat(osp.join(root, 'test.mat'))
        self.data = data['X'].astype(np.float32)#.reshape((-1, 28, 28, 1))
        self.label = data['Y'].astype(int).squeeze()
        self.data = self.data[self.label != 0]
        self.label = self.label[self.label != 0]
#        self.label -= 1 # why do this?
#        self.size = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.label[item]

        if self.transform is not None:
         data = self.transform(data)

        data = torch.from_numpy(data)
#        data = torch.from_numpy(np.array(data, copy=False))
#        data = data.view(self.size, self.size, 1)
#        data = data.transpose(0, 1).transpose(0, 2).contiguous()

        if self.transform2 is not None:
            data = self.transform2(data)

        return data, target

# Sampling mnist data for outlier experiment
class outlier2(data.Dataset):
    """Custom Dataset loader for outlier exp"""
    def __init__(self, root, transform=None, transform2=None, size=28, train=True, proportions=(0.1*np.ones(10))):
        assert(np.all(proportions>=0) and abs(np.sum(proportions)-1)<1e-10)
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        if train:
            data = sio.loadmat(osp.join(root, 'mnist.mat'))
        else:
            data = sio.loadmat(osp.join(root, 'test.mat'))
        self.data = data['X'].astype(np.float32)#.reshape((-1, 28, 28, 1))
        self.label = data['Y'].astype(int).squeeze()

        choiceInds = []
        for i in range(10):
          toChange = np.argwhere(self.label==i)
          toChange = toChange.reshape((len(toChange)))
          choices = np.random.choice(len(toChange),int(proportions[i]*len(self.label)))
          choiceInds.append(toChange[choices])
        allChoices = np.concatenate(choiceInds)
        np.random.shuffle(allChoices)

        self.data = self.data[allChoices]
        self.label = self.label[allChoices]
#        self.label -= 1 # why do this?
#        self.size = size
        print self.data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.label[item]

        if self.transform is not None:
         data = self.transform(data)

        data = torch.from_numpy(data)
#        data = torch.from_numpy(np.array(data, copy=False))
#        data = data.view(self.size, self.size, 1)
#        data = data.transpose(0, 1).transpose(0, 2).contiguous()

        if self.transform2 is not None:
            data = self.transform2(data)

        return data, target


# Sampling MoG - complete
def generate_data(num_mode, except_num, radius=1, center=(0, 0), sigma=0.01, num_data_per_class=1600000):
    total_data = {}

    t = np.linspace(0, 2 * np.pi, num_mode+1)
    x = np.cos(t) * radius + center[0]
    y = np.sin(t) * radius + center[1]

    modes = np.vstack([x, y]).T

    for idx, mode in enumerate(modes[except_num:]):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        total_data[idx] = np.vstack([x, y]).T

    all_points = np.vstack([values for values in total_data.values()])

    data = torch.from_numpy(all_points).float()
    dataset = TensorDataset(data, data)

    return dataset

# Sampling MoG - complete
def generate_data_uniform(radius=3):
    total_data = {}

    a = np.linspace(-radius,radius,1000)
    x = np.tile(a,1000)
    y = np.tile(a,(1000,1)).T.reshape(1*10**6)
    all_points = np.vstack([x, y]).T

    data = torch.from_numpy(all_points).float()
    dataset = TensorDataset(data, data)

    return dataset

# Sampling MoG - single batch
def generate_data_single_batch(num_mode, except_num, batch_size=512, radius=1, center=(0, 0), sigma=0.01):
    total_data = {}
    num_data_per_class = int(np.ceil(float(batch_size) / (num_mode-except_num+1)))

    t = np.linspace(0, 2 * np.pi, num_mode+1)
    x = np.cos(t) * radius + center[0]
    y = np.sin(t) * radius + center[1]

    modes = np.vstack([x, y]).T

    for idx, mode in enumerate(modes[except_num:]):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        total_data[idx] = np.vstack([x, y]).T

    all_points = np.vstack([values for values in total_data.values()])
    all_points = np.random.permutation(all_points)[0:batch_size]

    data = torch.from_numpy(all_points).float()

    return data

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

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias.data is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)

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
