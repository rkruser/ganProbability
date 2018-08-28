import numpy as np
import scipy.io as sio # for mats
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data

mnist = dset.MNIST(root='/vulcan/scratch/krusinga/mnist', download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

#print np.shape(mnist[0][0].numpy())
#print mnist[0][1]

allIms = np.empty((len(mnist),1,28,28))
allLabels = np.empty(len(mnist))
for i in range(len(mnist)):
  allIms[i,:,:,:] = mnist[i][0].numpy()
  allLabels[i] = mnist[i][1]

sio.savemat('mnist.mat',{'X':allIms,'Y':allLabels},appendmat=False)
