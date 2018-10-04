import numpy as np
from sklearn.decomposition import PCA
import pickle
from scipy.io import loadmat
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from code.total.loaders.MatLoaders import MatLoader



def main():
  mnistMat = loadmat(open('/vulcan/scratch/krusinga/mnist/mnist28.mat'))
  mnistX = mnistMat['Xtrain']
  mnistY = mnistMat['Ytrain']

  clf = PCA(n_components=2)
  mnistX = np.reshape(mnistX, (len(mnistX),784))
  print "Fitting clf"
  clf.fit(mnistX)
  projected = clf.transform(mnistX)
  plt.plot(projected)
  plt.savefig("scripts/mnistPlot.png", format='png')
  pickle.dump(clf, open('/vulcan/scratch/krusinga/mnist/mnistPCA.pickle','w'))



if __name__=='__main__':
  main()



