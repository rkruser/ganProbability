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
from torch.autograd import Variable

from mlworkflow import FileManager
from mlworkflow.defaults import imageArray
from code.total.getOperators import mapDict as Modules
from code.total.models.nnModels import NthArgWrapper
#import code.total.getOperators
#Modules = getOperators.mapDict


def extractProbabilityStats(probs, ims, labels, take=36):#, plot=False, save=True):
    sortedArgs = np.argsort(probs)
    sortedLabels = labels[sortedArgs]
    sortedProbs = probs[sortedArgs]
    sortedIms = ims[sortedArgs]
    topBottomAll = [None, sortedIms[:take], sortedIms[-take:]]
    labelAccs = []
    imAccs = [topBottomAll]
    for lbl in np.unique(labels):
        mask = (sortedLabels==lbl)
        labelAccs.append([lbl, np.cumsum(mask), sortedProbs[mask]])
        imAccs.append([lbl, sortedIms[mask][:take], sortedIms[mask][-take:]])
    return labelAccs, imAccs, sortedProbs

def plotHistograms(accs, suffix='',sortedProbs=None):
    for d in accs:
        lbl, cumul, probs = d[0], d[1], d[2]
        plt.plot(range(len(cumul)), cumul, label='Label {}'.format(lbl))
        plt.legend(loc='topleft')
    plt.title("Class accumulations")
    plt.xlabel("Sorted point index")
    plt.ylabel("Number of preceding points with class label")
    plt.savefig('./experiments/e65/analysis/cumulativeClassHistogram_{}.png'.format(suffix))
    plt.close()

    if sortedProbs is not None:
        plt.hist(sortedProbs, bins=100)
        plt.savefig('./experiments/e65/analysis/totalProbHistogram_{}.png'.format(suffix))
        plt.close()

    for d in accs:
        lbl, cumul, probs = d[0], d[1], d[2]
        plt.hist(probs, bins=100)
        plt.title("Class {} logprob histogram".format(lbl))
        plt.xlabel("Log probability")
        plt.ylabel("Number of points")
        plt.savefig('./experiments/e65/analysis/probHistClass_{0}_{1}.png'.format(suffix,lbl))
        plt.close()

def plotTopBottom(imAccs, suffix=''):
    for imList in imAccs:
        lbl, bottom, top = imList[0], imList[1], imList[2]
        bottom = np.transpose(bottom,(0,2,3,1))*0.5+0.5
        top = np.transpose(top,(0,2,3,1))*0.5+0.5
        imageArray({'images':bottom},'./experiments/e65/analysis/bottom_{0}_{1}.png'.format(suffix,str(lbl)))
        imageArray({'images':top},'./experiments/e65/analysis/top_{0}_{1}.png'.format(suffix,str(lbl)))
        plt.close()
 

       






def mnistNumerical():
    #prefix = '/fs/vulcan-scratch/krusinga/projects/ganProbability/generated/canonical/DCGAN/MNIST/size32_3'
    masterpath = '/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'
    cuda = True
    currentExperiment = 65
    files = FileManager(masterpath, currentExperiment = currentExperiment)

    mnist32loader = Modules['MNISTSize32Cols1']({'name':'MNIST32', 'fileHandler':files, 'dependencies':[]},{})
    mnist32 = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)
    Xtest = np.concatenate([mnist32.X, mnist32.X, mnist32.X], axis=1)
    Ytest = mnist32.Y
    print Xtest.shape
    print Ytest.shape

    probsPixel = None
    probsDeep = None
    if not osp.exists('./generated/e65/data/probsPixel.pickle'):
        numericalPixelRegressor = Modules['RegressorSize32Col3']({'name':'NumericalPixels',
                                                                  'fileHandler':files,
                                                                  'dependencies':[]},
                                                                 {'netPkey':'reg_mnist_32_3_numerical_pixel','cuda':cuda})
        numericalDeepRegressor = Modules['FeatRegressor10']({'name':'NumericalDeep',
                                                                  'fileHandler':files,
                                                                  'dependencies':[]},
                                                                 {'netPkey':'reg_mnist_32_3_numerical_deep','cuda':cuda})
        lenetModule = Modules['LenetSize32Cols3']({'name':'lenetMnist32',
                                                'fileHandler':files,
                                                'dependencies':[]},
                                                {'lenetKey':'deep_mnist_32_3','cuda':cuda})
        lenet = NthArgWrapper(lenetModule.getModel(), 1)
        
        XtestRun = Variable(torch.Tensor(Xtest))
        if cuda:
            XtestRun = XtestRun.cuda()
        XtestEmbedded = lenet(XtestRun)
        print "Getting pixel probs"
        probsPixel = numericalPixelRegressor.netP(XtestRun).data.cpu().numpy()
        print "Shape:",probsPixel.shape
        print "Getting deep probs"
        probsDeep = numericalDeepRegressor.netP(XtestEmbedded).data.cpu().numpy()
        print "Shape:",probsDeep.shape


        print "Dumping pixel"
        pickle.dump(probsPixel,open('./generated/e65/data/probsPixel.pickle','w')) 
        print "Dumping deep"
        pickle.dump(probsDeep,open('./generated/e65/data/probsDeep.pickle','w')) 

    else:
        probsPixel = pickle.load(open('./generated/e65/data/probsPixel.pickle','r')) 
        probsDeep = pickle.load(open('./generated/e65/data/probsDeep.pickle','r')) 
       

    
    pixelStats = extractProbabilityStats(probsPixel, Xtest, Ytest)
    deepStats = extractProbabilityStats(probsDeep, Xtest, Ytest)
    
    plotHistograms(pixelStats[0],suffix='pixel', sortedProbs = pixelStats[2])
    plotHistograms(deepStats[0], suffix='deep', sortedProbs = deepStats[2])
#    plotTopBottom(pixelStats[1], suffix='pixel')
#    plotTopBottom(deepStats[1], suffix='deep')
    



if __name__=='__main__':
    mnistNumerical()



























