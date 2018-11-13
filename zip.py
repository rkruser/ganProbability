import argparse
from scipy.io import loadmat, savemat
import numpy as np
from os.path import join

def zipTrainTest(train, test, out):
	m1 = loadmat(train)
	m2 = loadmat(test)
	m = {
		'Xtrain':m1['X'],
		'Ytrain':m1['Y'],
		'Xtest':m2['X'],
		'Ytest':m2['Y']
	}
	savemat(out, m)

def zipSamples(samplePrefix, nfiles, out):
	loadedMats = []
	for i in range(nfiles):
		loadedMats.append(loadmat(samplePrefix+'_'+str(i)+'.mat'))

	allSamples = {
	  'images': np.concatenate([loadedMats[i]['images'] for i in range(nfiles)]),
	  'jacob': np.concatenate([loadedMats[i]['jacob'] for i in range(nfiles)]),
	  'codes': np.concatenate([loadedMats[i]['codes'] for i in range(nfiles)]),
	  'probs': np.concatenate([loadedMats[i]['probs'].flatten() for i in range(nfiles)])
	}
	savemat(out, allSamples)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--zipsamples', action='store_true', help='Zip together samples in a folder')
	parser.add_argument('--zipTrainTest', action='store_true', help='Zip together train and test points in separate files')
	parser.add_argument('--train', type=str, default=None, help='Train file')
	parser.add_argument('--test', type=str, default=None, help='Test file')
	parser.add_argument('--out', type=str, default='allSamples.mat', help='Output file name')
	parser.add_argument('--samplePrefix', type=str, default=None, help='Prefix of sample files')
	parser.add_argument('--nfiles', type=int, default=None, help='Number of sample files')

	opt = parser.parse_args()
	if opt.zipsamples:
		zipSamples(opt.samplePrefix, opt.nfiles, opt.out)
	if opt.zipTrainTest:
		zipTrainTest(opt.train, opt.test, opt.out)


if __name__ == '__main__':
	main()
