from __future__ import division

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils
from scipy.io import loadmat, savemat
import numpy as np
import argparse

from loaders import locations


def extractPercentileStats(trainProbs, testProbs):
    sortedInds = np.argsort(np.concatenate([trainProbs, testProbs]))
    trainTest = np.concatenate([np.ones(len(trainProbs), dtype='int'),np.zeros(len(testProbs),dtype='int')])
    trainTest = trainTest[sortedInds]
    cumulative = np.cumsum(trainTest)
    testPercentiles = cumulative[trainTest==0]/float(len(trainProbs))
    return testPercentiles

   



def heatmap(pre, hmap):
	grid = hmap['grid']


def percentilePlot():
	pass

def hiLow(prefix, gSample, dataset):
	# Sort probs into ims, and then
	mGrid = vutils.make_grid(ims, normalize=True, scale_each=True)
	mGrid = mGrid.numpy()
	plt.imshow(mGrid)


def probHistogram():
	pass

def noiseInfluence():
	pass

def epsInfluence():
	pass

def fuzzingInfluence():
	pass




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--plot', required=True, type=str, default='hiLow', help='Type of plot')
	parser.add_argument('--heatmapMat', type=str, default=None, help='Mat with GAN image and probability samples')
	parser.add_argument('--dataset', type=str, default=None, help='Dataset probabilities were computed on')
	parser.add_argument('--probsMat', type=str, default=None, help='Mat with probabilities of some dataset')
	parser.add_argument('--trainMat', type=str, default=None, help='Mat with probabilities on train data')
	parser.add_argument('--testMat', type=str, default=None, help='Mat with probabilities on test data')None
	parser.add_argument('--savePrefix', required=True, type=str, default='plot', help='Prefix path and fname for the plot')

	opt = parser.parse_args()

	if opt.plot == 'heatmap':
		hmap = loadmat(opt.heatmapMat)
		heatmap(opt.savePrefix, hmap)
	elif opt.plot == 'percentile':
		train = loadmat(opt.trainMat)
		test = loadmat(opt.testMat)
		percentilePlot(opt.savePrefix,train,test)
	elif opt.plot == 'hiLow':
		probs = loadmat(opt.probsMat)
		if opt.dataset is not None:
			dataset = loadmat(locations[opt.dataset])
		else:
			dataset = None
		hiLow(opt.savePrefix, probs, dataset)
	elif opt.plot == 'probHistogram':
		probs = loadmat(opt.probsMat)
		if opt.dataset is not None:
			dataset = loadmat(locations[opt.dataset])
		else:
			dataset = None
		probHistogram(opt.savePrefix, probs, dataset)




if __name__=='__main__':
	main()