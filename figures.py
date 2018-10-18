from __future__ import division

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
from code.total.loaders.MatLoaders import MatLoader


def extractProbabilityStats(probs, ims, labels, take=36):#, plot=False, save=True):
    sortedArgs = np.argsort(probs)
    sortedLabels = labels[sortedArgs]
    sortedProbs = probs[sortedArgs]
    sortedIms = ims[sortedArgs]
    topBottomAll = [None, sortedIms[:take], sortedIms[-take:]]
    labelAccs = []
    imAccs = [topBottomAll]
#    for lbl in np.unique(labels):
#        mask = (sortedLabels==lbl)
#        labelAccs.append([lbl, np.cumsum(mask), sortedProbs[mask]])
#        imAccs.append([lbl, sortedIms[mask][:take], sortedIms[mask][-take:]])
    return labelAccs, imAccs, sortedProbs

def extractPercentileStats(trainProbs, testProbs):
    sortedInds = np.argsort(np.concatenate([trainProbs, testProbs]))
    trainTest = np.concatenate([np.ones(len(trainProbs), dtype='int'),np.zeros(len(testProbs),dtype='int')])
    trainTest = trainTest[sortedInds]
    cumulative = np.cumsum(trainTest)
    testPercentiles = cumulative[trainTest==0]/float(len(trainProbs))
    return testPercentiles



def plotHistograms(accs, path='', suffix='',sortedProbs=None):
    for d in accs:
        lbl, cumul, probs = d[0], d[1], d[2]
        plt.plot(range(len(cumul)), cumul, label='Label {}'.format(lbl))
        plt.legend(loc='topleft')
    plt.title("Class accumulations")
    plt.xlabel("Sorted point index")
    plt.ylabel("Number of preceding points with class label")
    plt.savefig(osp.join(path,'cumulativeClassHistogram_{}.png'.format(suffix)))
    plt.close()

    if sortedProbs is not None:
        plt.hist(sortedProbs, bins=100)
        plt.savefig(osp.join(path,'totalProbHistogram_{}.png'.format(suffix)))
        plt.close()

    for d in accs:
        lbl, cumul, probs = d[0], d[1], d[2]
        plt.hist(probs, bins=100)
        plt.title("Class {} logprob histogram".format(lbl))
        plt.xlabel("Log probability")
        plt.ylabel("Number of points")
        plt.savefig(osp.join(path,'probHistClass_{0}_{1}.png'.format(suffix,lbl)))
        plt.close()

def plotTopBottom(imAccs, path='', suffix=''):
    for imList in imAccs:
        lbl, bottom, top = imList[0], imList[1], imList[2]
        bottom = np.transpose(bottom,(0,2,3,1))*0.5+0.5
        top = np.transpose(top,(0,2,3,1))*0.5+0.5
        imageArray({'images':bottom},osp.join(path,'bottom_{0}_{1}.png'.format(suffix,str(lbl))))
        imageArray({'images':top},osp.join(path,'top_{0}_{1}.png'.format(suffix,str(lbl))))
        plt.close()
 

       
# data is a torch tensor, pre-cuda
# network is a torch network
def runHuge(network, data, cuda=True):
    chunks = torch.chunk(data,100)
    outChunks = []
    for ch in chunks:
        chVar = Variable(ch)
        if cuda:
            chVar = chVar.cuda()
        result = network(chVar)
        outChunks.append(result.data.cpu())
    return torch.cat(outChunks)



# def mnistNumericalSave():
#     #prefix = '/fs/vulcan-scratch/krusinga/projects/ganProbability/generated/canonical/DCGAN/MNIST/size32_3'
#     masterpath = '/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'
#     cuda = True
#     currentExperiment = 65
#     files = FileManager(masterpath, currentExperiment = currentExperiment)

#     print "Loading datasets"
#     mnist32loader = Modules['MNISTSize32Cols1']({'name':'MNIST32', 'fileHandler':files, 'dependencies':[]},{})
#     mnist32train = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='train', returnLabel = True)
#     mnist32 = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)
#     print "Done loading datasets"
#     Xtrain = np.concatenate([mnist32train.X, mnist32train.X, mnist32train.X], axis=1)
#     Ytrain = mnist32train.Y
#     Xtest = np.concatenate([mnist32.X, mnist32.X, mnist32.X], axis=1)
#     Ytest = mnist32.Y
#     print Xtest.shape
#     print Ytest.shape

#     probsPixel = None
#     probsDeep = None

# #    if not osp.exists('./generated/e65/data/probsPixel.pickle'):
#     numericalPixelRegressor = Modules['RegressorSize32Col3']({'name':'NumericalPixels',
#                                                               'fileHandler':files,
#                                                               'dependencies':[]},
#                                                              {'netPkey':'reg_mnist_32_3_numerical_pixel','cuda':cuda})
#     numericalDeepRegressor = Modules['FeatRegressor10']({'name':'NumericalDeep',
#                                                               'fileHandler':files,
#                                                               'dependencies':[]},
#                                                              {'netPkey':'reg_mnist_32_3_numerical_deep','cuda':cuda})
#     lenetModule = Modules['LenetSize32Cols3']({'name':'lenetMnist32',
#                                             'fileHandler':files,
#                                             'dependencies':[]},
#                                             {'lenetKey':'deep_mnist_32_3','cuda':cuda})
#     lenet = NthArgWrapper(lenetModule.getModel(), 1)
    
#     XtrainRun = torch.Tensor(Xtrain)
#     XtestRun = torch.Tensor(Xtest)
#     # if cuda:
#     #     XtrainRun = XtrainRun.cuda()
#     #     XtestRun = XtestRun.cuda()
# #    XtrainEmbedded = lenet(XtrainRun)
# #    XtestEmbedded = lenet(XtestRun)
#     lenet.eval()
#     numericalPixelRegressor.netP.eval()
#     numericalDeepRegressor.netP.eval()

#     XtrainEmbedded = runHuge(lenet, XtrainRun)
#     XtestEmbedded = runHuge(lenet, XtestRun)
#     print "Getting pixel probs"
#     probsPixelTrain = runHuge(numericalPixelRegressor.netP, XtrainRun)
#     probsPixel = runHuge(numericalPixelRegressor.netP, XtestRun)
#     print "Shape:",probsPixel.shape, probsPixelTrain.shape
#     print "Getting deep probs"
#     probsDeepTrain = runHuge(numericalDeepRegressor.netP, XtrainEmbedded)
#     probsDeep = runHuge(numericalDeepRegressor.netP, XtestEmbedded)
#     print "Shape:",probsDeep.shape, probsDeepTrain.shape


#     print "Dumping pixel"
#     pickle.dump(probsPixel.numpy(),open('./generated/e65/data/probsPixel.pickle','w')) 
#     pickle.dump(probsPixelTrain.numpy(),open('./generated/e65/data/probsPixelTrain.pickle','w'))
#     print "Dumping deep"
#     pickle.dump(probsDeep.numpy(),open('./generated/e65/data/probsDeep.pickle','w')) 
#     pickle.dump(probsDeepTrain.numpy(),open('./generated/e65/data/probsDeepTrain.pickle','w'))

# def mnistNumericalPlot():
#     probsPixel = pickle.load(open('./generated/e65/data/probsPixel.pickle','r')) 
#     probsPixelTrain = pickle.load(open('./generated/e65/data/probsPixelTrain.pickle','r'))
#     probsDeep = pickle.load(open('./generated/e65/data/probsDeep.pickle','r')) 
#     probsDeepTrain = pickle.load(open('./generated/e65/data/probsDeepTrain.pickle','r'))
    
# #    pixelStats = extractProbabilityStats(probsPixel, Xtest, Ytest)
# #    deepStats = extractProbabilityStats(probsDeep, Xtest, Ytest)
#     pixelPercentiles = extractPercentileStats(probsPixelTrain, probsPixel)
#     deepPercentiles = extractPercentileStats(probsDeepTrain, probsDeep)
#     return pixelPercentiles, deepPercentiles
    
#    plotHistograms(pixelStats[0],path = './experiments/e65/analysis/', suffix='pixel', sortedProbs = pixelStats[2])
#    plotHistograms(deepStats[0], path = './experiments/e65/analysis/', suffix='deep', sortedProbs = deepStats[2])
#    plotTopBottom(pixelStats[1], path='./experiments/e65/analysis/', suffix='pixel')
#    plotTopBottom(deepStats[1], path='./experiments/e65/analysis/', suffix='deep')
    

def GetProbabilities(Xtrain, Xtest, deepNet, pixelProbNet, deepProbNet, savefolder, saveprefix):
        # XtestRun = Variable(torch.Tensor(Xtest))
    # if cuda:
    #     XtestRun = XtestRun.cuda()
    # XtestEmbedded = lenet(XtestRun)
    print "Set networks to eval"
    deepNet.eval()
    pixelProbNet.eval()
    deepProbNet.eval()

    print "Run deep features on xtrain and xtest"
    XtrainEmbedded = runHuge(deepNet, Xtrain)
    XtestEmbedded = runHuge(deepNet, Xtest)

    print "Run pixel clf"
    probsPixelTrain = runHuge(pixelProbNet, Xtrain)
    probsPixelTest = runHuge(pixelProbNet, Xtest)

    print "Run deep clf"
    probsDeepTrain = runHuge(deepProbNet, XtrainEmbedded)
    probsDeepTest = runHuge(deepProbNet, XtestEmbedded)

    print "Dumping pixel"
    pickle.dump(probsPixelTest.numpy(),open(osp.join(savefolder,saveprefix+'_probsPixelTest.pickle'),'w')) 
    pickle.dump(probsPixelTrain.numpy(),open(osp.join(savefolder,saveprefix+'_probsPixelTrain.pickle'),'w')) 

    print "Dumping deep"
    pickle.dump(probsDeepTest.numpy(),open(osp.join(savefolder,saveprefix+'_probsDeepTest.pickle'),'w')) 
    pickle.dump(probsDeepTrain.numpy(),open(osp.join(savefolder,saveprefix+'_probsDeepTrain.pickle'),'w'))

    return probsPixelTrain.numpy(), probsPixelTest.numpy(), probsDeepTrain.numpy(), probsDeepTest.numpy()


def GetPlots(loadfolder, savefolder, trainprefix, testprefix, saveprefix):
    pixelTrain = pickle.load(open(osp.join(loadfolder,trainprefix+'_probsPixelTrain.pickle'),'r')) 
    pixelTest = pickle.load(open(osp.join(loadfolder,testprefix+'_probsPixelTest.pickle'),'r')) 

    deepTrain = pickle.load(open(osp.join(loadfolder,trainprefix+'_probsDeepTrain.pickle'),'r'))
    deepTest = pickle.load(open(osp.join(loadfolder,testprefix+'_probsDeepTest.pickle'),'r'))

    pixelPercentiles = 100*extractPercentileStats(pixelTrain, pixelTest)
    deepPercentiles = 100*extractPercentileStats(deepTrain, deepTest)

    fig, ax = plt.subplots(1)
    pixelCounts, pixelBins = np.histogram(pixelPercentiles,bins=100)
    deepCounts, deepBins = np.histogram(deepPercentiles,bins=100)

#    pixelCounts = pixelCounts/np.trapz(pixelCounts,x=pixelBins[:len(pixelCounts)])#float(np.sum(pixelCounts))
#    deepCounts = deepCounts/np.trapz(deepCounts, x=deepBins[:len(deepCounts)])#float(np.sum(deepCounts))

    ax.plot(pixelBins[:len(pixelCounts)],pixelCounts,'r.-',label='Pixel Regressor')
#    plt.savefig(osp.join(savefolder,saveprefix+'_pixelHist.png'))

    ax.plot(deepBins[:len(deepCounts)],deepCounts,'b.-', label='Deep Feature Regressor')
 #   ax.set_ylim(ymin=0,ymax=11)
#    ax.set_ylim(ymin=0,ymax=700)
    ax.legend()

    return fig, ax
#    ax.set_ylim(ymin=0,ymax=0.07)
#    plt.savefig(osp.join(savefolder,saveprefix+'_deepHist.png'))

def GetHists(loadfolder, savefolder, trainprefix, testprefix, saveprefix):
    pixelTrain = pickle.load(open(osp.join(loadfolder,trainprefix+'_probsPixelTrain.pickle'),'r')) 
    pixelTest = pickle.load(open(osp.join(loadfolder,testprefix+'_probsPixelTest.pickle'),'r')) 

    deepTrain = pickle.load(open(osp.join(loadfolder,trainprefix+'_probsDeepTrain.pickle'),'r'))
    deepTest = pickle.load(open(osp.join(loadfolder,testprefix+'_probsDeepTest.pickle'),'r'))

    pixelPercentiles = 100*extractPercentileStats(pixelTrain, pixelTest)
    deepPercentiles = 100*extractPercentileStats(deepTrain, deepTest)

    return pixelPercentiles, deepPercentiles

#    fig, ax = plt.subplots(1)
#j    pixelCounts, pixelBins = np.histogram(pixelPercentiles,bins=100)
#    deepCounts, deepBins = np.histogram(deepPercentiles,bins=100)

#    return pixelCounts, pixelBins, deepCounts, deepBins

#    pixelCounts = pixelCounts/np.trapz(pixelCounts,x=pixelBins[:len(pixelCounts)])#float(np.sum(pixelCounts))
#    deepCounts = deepCounts/np.trapz(deepCounts, x=deepBins[:len(deepCounts)])#float(np.sum(deepCounts))

#    ax.hist(pixelCounts, bins=pixelBins[:len(pixelCounts)])#,'r.-',label='Pixel Regressor')
#    plt.savefig(osp.join(savefolder,saveprefix+'_pixelHist.png'))

#    ax.hist(deepBins[:len(deepCounts)],deepCounts,'b.-', label='Deep Feature Regressor')
 #   ax.set_ylim(ymin=0,ymax=11)
#    ax.set_ylim(ymin=0,ymax=700)
    #ax.legend()

#    return fig, ax


def mnistNumerical(cuda=True, currentExperiment=68, masterpath='/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'):
    savefolder='./generated/e{0}/data'.format(currentExperiment)
    saveprefix = 'mnistNumerical'
    files = FileManager(masterpath, currentExperiment = currentExperiment)

    print "Loading datasets"
    mnist32loader = Modules['MNISTSize32Cols1']({'name':'MNIST32', 'fileHandler':files, 'dependencies':[]},{})
    mnist32train = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='train', returnLabel = True)
    mnist32 = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)
    print "Done loading datasets"
    Xtrain = np.concatenate([mnist32train.X, mnist32train.X, mnist32train.X], axis=1)
    Ytrain = mnist32train.Y
    Xtest = np.concatenate([mnist32.X, mnist32.X, mnist32.X], axis=1)
    Ytest = mnist32.Y

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
    
    Xtrain = torch.Tensor(Xtrain)
    Xtest = torch.Tensor(Xtest)

    GetProbabilities(Xtrain, Xtest, lenet, numericalPixelRegressor.netP, numericalDeepRegressor.netP, savefolder, saveprefix)
#    GetPlots(savefolder, 'mnistNumerical', 'mnistNumerical', 'mnistTrain_mnistTest')

def cifarNumerical(cuda=True, currentExperiment=68, masterpath='/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'):
    savefolder='./generated/e{0}/data'.format(currentExperiment)
    saveprefix = 'cifarNumerical'
    files = FileManager(masterpath, currentExperiment = currentExperiment)

    cifar32loader = Modules['CFAR10Size32Cols3']({'name':'CIFAR10', 'fileHandler':files, 'dependencies':[]},{})
    cifar32train = cifar32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='train', returnLabel = True)
    cifar32test = cifar32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)

    Xtrain = torch.Tensor(cifar32train.X)
    Ytrain = torch.Tensor(cifar32test.Y)
    Xtest = torch.Tensor(cifar32test.X)
    Ytest = torch.Tensor(cifar32test.Y)

    numericalPixelRegressor = Modules['RegressorSize32Col3']({'name':'NumericalPixels',
                                                              'fileHandler':files,
                                                              'dependencies':[]},
                                                             {'netPkey':'reg_cifar10_32_3_numerical_pixel','cuda':cuda})
    numericalDeepRegressor = Modules['FeatRegressor10']({'name':'NumericalDeep',
                                                              'fileHandler':files,
                                                              'dependencies':[]},
                                                             {'netPkey':'reg_cifar10_32_3_numerical_deep','cuda':cuda})
    lenetModule = Modules['LenetSize32Cols3']({'name':'lenetCifar32',
                                            'fileHandler':files,
                                            'dependencies':[]},
                                            {'lenetKey':'deep_cifar10_32_3','cuda':cuda})
    lenet = NthArgWrapper(lenetModule.getModel(), 1)


    GetProbabilities(Xtrain, Xtest, lenet, numericalPixelRegressor.netP, numericalDeepRegressor.netP, savefolder, saveprefix)
#    GetPlots(savefolder, 'cifarNumerical', 'cifarNumerical', 'cifarTrain_cifarTest')


def mnistOnCifar(cuda=True, currentExperiment=68, masterpath='/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'):
    savefolder='./generated/e{0}/data'.format(currentExperiment)
    saveprefix = 'mnistNumericalOnCifar'
    files = FileManager(masterpath, currentExperiment = currentExperiment)

    cifar32loader = Modules['CFAR10Size32Cols3']({'name':'CIFAR10', 'fileHandler':files, 'dependencies':[]},{})
    cifar32train = cifar32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='train', returnLabel = True)
    cifar32test = cifar32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)

    Xtrain = torch.Tensor(cifar32train.X)
    Ytrain = torch.Tensor(cifar32test.Y)
    Xtest = torch.Tensor(cifar32test.X)
    Ytest = torch.Tensor(cifar32test.Y)

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
    
    Xtrain = torch.Tensor(Xtrain)
    Xtest = torch.Tensor(Xtest)

    GetProbabilities(Xtrain, Xtest, lenet, numericalPixelRegressor.netP, numericalDeepRegressor.netP, savefolder, saveprefix)
#    GetPlots(savefolder, 'mnistNumerical', 'mnistNumericalOnCifar', 'mnistTrain_cifarTest')

def mnistOnOmniglot(cuda=True, currentExperiment=69, masterpath='/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'):
    savefolder='./generated/e{0}/data'.format(currentExperiment)
    saveprefix = 'mnistNumericalOnOmniglot'
    files = FileManager(masterpath, currentExperiment = currentExperiment)

    dataTrain = MatLoader(files.getFilePath('japanese_hiragana_32'), outShape = (3,32,32), returnLabel=True, mode='train')
    dataTest = MatLoader(files.getFilePath('japanese_hiragana_32'), outShape = (3,32,32), returnLabel=True, mode='test')

    Xtrain = torch.Tensor(dataTrain.X)
    Ytrain = torch.Tensor(dataTrain.Y)
    Xtest = torch.Tensor(dataTest.X)
    Ytest = torch.Tensor(dataTest.Y)

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
    
    Xtrain = torch.Tensor(Xtrain)
    Xtest = torch.Tensor(Xtest)

    pixTr, pixTe, deTr, deTe = GetProbabilities(Xtrain, Xtest, lenet, numericalPixelRegressor.netP, numericalDeepRegressor.netP, savefolder, saveprefix)
#    GetPlots(savefolder, 'mnistNumerical', 'mnistNumericalOnCifar', 'mnistTrain_cifarTest')

    pixelStats = extractProbabilityStats(pixTr, Xtrain, Ytrain)
    deepStats = extractProbabilityStats(deTr, Xtrain, Ytrain)
    
#    plotHistograms(pixelStats[0],path = './experiments/e65/analysis/', suffix='pixel', sortedProbs = pixelStats[2])
#    plotHistograms(deepStats[0], path = './experiments/e65/analysis/', suffix='deep', sortedProbs = deepStats[2])
    plotTopBottom(pixelStats[1], path='./experiments/e69/analysis/', suffix='pixel')
    plotTopBottom(deepStats[1], path='./experiments/e69/analysis/', suffix='deep')




def cifarOnMnist(cuda=True, currentExperiment=68, masterpath='/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'):
    savefolder='./generated/e{0}/data'.format(currentExperiment)
    saveprefix = 'cifarNumericalOnMnist'
    files = FileManager(masterpath, currentExperiment = currentExperiment)

    print "Loading datasets"
    mnist32loader = Modules['MNISTSize32Cols1']({'name':'MNIST32', 'fileHandler':files, 'dependencies':[]},{})
    mnist32train = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='train', returnLabel = True)
    mnist32 = mnist32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)
    print "Done loading datasets"
    Xtrain = torch.Tensor(np.concatenate([mnist32train.X, mnist32train.X, mnist32train.X], axis=1))
    Ytrain = mnist32train.Y
    Xtest = torch.Tensor(np.concatenate([mnist32.X, mnist32.X, mnist32.X], axis=1))
    Ytest = mnist32.Y


    numericalPixelRegressor = Modules['RegressorSize32Col3']({'name':'NumericalPixels',
                                                              'fileHandler':files,
                                                              'dependencies':[]},
                                                             {'netPkey':'reg_cifar10_32_3_numerical_pixel','cuda':cuda})
    numericalDeepRegressor = Modules['FeatRegressor10']({'name':'NumericalDeep',
                                                              'fileHandler':files,
                                                              'dependencies':[]},
                                                             {'netPkey':'reg_cifar10_32_3_numerical_deep','cuda':cuda})
    lenetModule = Modules['LenetSize32Cols3']({'name':'lenetCifar32',
                                            'fileHandler':files,
                                            'dependencies':[]},
                                            {'lenetKey':'deep_cifar10_32_3','cuda':cuda})
    lenet = NthArgWrapper(lenetModule.getModel(), 1)


    GetProbabilities(Xtrain, Xtest, lenet, numericalPixelRegressor.netP, numericalDeepRegressor.netP, savefolder, saveprefix)
#    GetPlots(savefolder, 'cifarNumerical', 'cifarNumericalOnMnist', 'cifarTrain_mnistTest')



# def cifar10Numerical():
#     #prefix = '/fs/vulcan-scratch/krusinga/projects/ganProbability/generated/canonical/DCGAN/MNIST/size32_3'
#     masterpath = '/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml'
#     cuda = True
#     currentExperiment = 65
#     files = FileManager(masterpath, currentExperiment = currentExperiment)

#     cifar32loader = Modules['CFAR10Size32Cols3']({'name':'CIFAR10', 'fileHandler':files, 'dependencies':[]},{})
#     cifar32train = cifar32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='train', returnLabel = True)
#     cifar32test = cifar32loader.getDataset(outShape = (3,32,32), distribution=None, labels=None, mode='test', returnLabel = True)
#     #Xtest = np.concatenate([cifar32.X, cifar32.X, cifar32.X], axis=1)
#     Xtrain = torch.Tensor(cifar32train.X)
#     Ytrain = torch.Tensor(cifar32test.Y)
#     Xtest = torch.Tensor(cifar32test.X)
#     Ytest = torch.Tensor(cifar32test.Y)
#     print Xtrain.shape, Xtest.shape
#     print Xtest.shape, Ytest.shape

# #    if not osp.exists('./generated/e67/data/probsPixel.pickle'):
#     numericalPixelRegressor = Modules['RegressorSize32Col3']({'name':'NumericalPixels',
#                                                               'fileHandler':files,
#                                                               'dependencies':[]},
#                                                              {'netPkey':'reg_cifar10_32_3_numerical_pixel','cuda':cuda})
#     numericalDeepRegressor = Modules['FeatRegressor10']({'name':'NumericalDeep',
#                                                               'fileHandler':files,
#                                                               'dependencies':[]},
#                                                              {'netPkey':'reg_cifar10_32_3_numerical_deep','cuda':cuda})
#     lenetModule = Modules['LenetSize32Cols3']({'name':'lenetCifar32',
#                                             'fileHandler':files,
#                                             'dependencies':[]},
#                                             {'lenetKey':'deep_cifar10_32_3','cuda':cuda})
#     lenet = NthArgWrapper(lenetModule.getModel(), 1)

#     lenet.eval()
#     numericalDeepRegressor.netP.eval()
#     numericalPixelRegressor.netP.eval()
    
#     # XtestRun = Variable(torch.Tensor(Xtest))
#     # if cuda:
#     #     XtestRun = XtestRun.cuda()
#     # XtestEmbedded = lenet(XtestRun)
#     XtrainEmbedded = runHuge(lenet, Xtrain)
#     XtestEmbedded = runHuge(lenet, Xtest)
#     print "Getting pixel probs"
#     probsPixelTrain = runHuge(numericalPixelRegressor.netP, Xtrain)
#     probsPixelTest = runHuge(numericalPixelRegressor.netP, Xtest)
#     print "Shape:",probsPixelTrain.shape
#     print "Getting deep probs"
#     probsDeepTest = runHuge(numericalDeepRegressor.netP, XtestEmbedded)
#     probsDeepTrain = runHuge(numericalDeepRegressor.netP, XtrainEmbedded)
#     print "Shape:",probsDeepTrain.shape


#     print "Dumping pixel"
#     pickle.dump(probsPixelTest,open('./generated/e67/data/probsPixelTest.pickle','w')) 
#     pickle.dump(probsPixelTrain,open('./generated/e67/data/probsPixelTrain.pickle','w')) 

#     print "Dumping deep"
#     pickle.dump(probsDeepTest,open('./generated/e67/data/probsDeepTest.pickle','w')) 
#     pickle.dump(probsDeepTrain,open('./generated/e67/data/probsDeepTrain.pickle','w')) 


# def cifar10numericalPlot():
#     probsPixelTest = pickle.load(open('./generated/e67/data/probsPixelTest.pickle','r')) 
#     probsPixelTrain = pickle.load(open('./generated/e67/data/probsPixelTrain.pickle','r')) 
#     probsDeepTest = pickle.load(open('./generated/e67/data/probsDeepTest.pickle','r')) 
#     probsDeepTrain = pickle.load(open('./generated/e67/data/probsDeepTrain.pickle','r')) 

#     pixelPercentiles = extractPercentileStats(probsPixelTrain, probsPixelTest)
#     deepPercentiles = extractPercentileStats(probsDeepTrain, probsDeepTest)
#     return pixelPercentiles, deepPercentiles
    
#    pixelStats = extractProbabilityStats(probsPixel, Xtest, Ytest)
#    deepStats = extractProbabilityStats(probsDeep, Xtest, Ytest)
    
    # plotHistograms(pixelStats[0],path='./experiments/e67/analysis', suffix='pixel', sortedProbs = pixelStats[2])
    # plotHistograms(deepStats[0], path='./experiments/e67/analysis', suffix='deep', sortedProbs = deepStats[2])
    # plotTopBottom(pixelStats[1], path='./experiments/e67/analysis/', suffix='pixel')
    # plotTopBottom(deepStats[1], path='./experiments/e67/analysis/', suffix='deep')



if __name__=='__main__':
    opt = int(sys.argv[1])
    if opt == 0:
        mnistNumerical()
    elif opt == 1:
        cifarNumerical()
    elif opt == 2:
        mnistOnCifar()
    elif opt == 3:
        cifarOnMnist()
    elif opt == 4:
      mnistOnOmniglot()

  
