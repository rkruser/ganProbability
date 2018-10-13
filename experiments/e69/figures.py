import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle
from PIL import Image
import os.path as osp
import sys
#sys.path.append('..')
#print os.listdir('.')
#print sys.path
#sys.path.append(osp.abspath('..'))
#import getOperators

import torch

from mlworkflow import FileManager
from code.total.getOperators import mapDict as Modules

#prefix = '/fs/vulcan-scratch/krusinga/projects/ganProbability/generated/canonical/DCGAN/MNIST/size32_3'
masterpath = '/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'
currentExperiment = 65
files = FileManager(masterpath, currentExperiment = currentExperiment)

mnist32loader = Modules['MNISTSize32Cols1']({'name':'MNIST32', 'fileHandler':files, 'dependencies':[]},{})
numericalPixelRegressor = Modules['RegressorSize32Col3']({'name':'NumericalPixels',
                                                          'fileHandler':files,
                                                          'dependencies':[]},
                                                         {'netPkey':'reg_mnist_32_3_numerical_pixel'})
numericalDeepRegressor = Modules['FeatRegressor10']({'name':'NumericalDeep',
                                                          'fileHandler':files,
                                                          'dependencies':[]},
                                                         {'netPkey':'reg_mnist_32_3_numerical_deep'})
lenet = Modules['LenetSize32Cols3']({'name':'lenetMnist32',
                                        'fileHandler':files,
                                        'dependencies':[]},
                                        {'lenetKey':'deep_mnist_32_3'})