# Extract the top and bottom so many images
# Transform to viewable format
import scipy.io as sio #use sio.loadmat and sio.savemat
import numpy as np
import argparse
import os.path as osp
import sys

parser = argparse.ArgumentParser()
#parser.add_argument('--matfile', type=str, default='./allSamples.mat', help='The matfile with all images and files sampled')
parser.add_argument('--take', type=int, default=100, help='The number of images to take from the top and bottom')
parser.add_argument('--matprefix', type=str, default='features_', help='The prefix to all mat file names')
parser.add_argument('--matfolder', type=str, default='./', help='The folder containing the mat files to be joined')
parser.add_argument('--numfiles', type=int, default=10, help='The number of mat files to join')

opt = parser.parse_args()

loadedMats = []
for i in range(opt.numfiles):
  fpath = osp.join(opt.matfolder, opt.matprefix+('%d.mat'%(i)))
  loadedMats.append(sio.loadmat(fpath))

allSamples = {
  'images': np.concatenate([loadedMats[i]['images'] for i in range(opt.numfiles)]),
  'jacob': np.concatenate([loadedMats[i]['jacob'] for i in range(opt.numfiles)]),
  'code': np.concatenate([loadedMats[i]['code'] for i in range(opt.numfiles)]),
  'prob': np.concatenate([loadedMats[i]['prob'].flatten() for i in range(opt.numfiles)])
}

sio.savemat(osp.join(opt.matfolder,'allSamples.mat'), allSamples)

#allSamples = sio.loadmat(opt.matfile)

# images, code, jacob, prob

# Need to transform images 
#sortedInds = np.argsort(allSamples['prob'].reshape((np.size(allSamples['prob']))))
sortedInds = np.argsort(allSamples['prob'])
topN = sortedInds[-opt.take:]
bottomN = sortedInds[:opt.take]

imTop = allSamples['images'][topN]
probTop = allSamples['prob'][topN]
jacobTop = allSamples['jacob'][topN]
codeTop = allSamples['code'][topN]
imTop = (imTop+1.0)*127.5
imTop = imTop.astype(np.uint8)
imTop = np.transpose(imTop,(2,3,1,0)) # Transpose to RGB format
# put number of samples in last dimension because
# that's hows matlab needs it for imshow

imBot = allSamples['images'][bottomN]
probBot = allSamples['prob'][bottomN]
jacobBot = allSamples['jacob'][bottomN]
codeBot = allSamples['code'][bottomN]
imBot = (imBot+1.0)*127.5
imBot = imBot.astype(np.uint8)
imBot = np.transpose(imBot,(2,3,1,0))

#print np.shape(imTop)
#print np.shape(imBot)
#print np.shape(jacobTop)
#print np.shape(probTop)
#print probTop
#print probBot

sio.savemat(osp.join(opt.matfolder,'topBottom%d.mat'%opt.take), 
    {
      'top':{
        'images':imTop,
        'jacob':jacobTop,
        'code':codeTop,
        'prob':probTop 
      },
     'bottom':{
        'images':imBot,
        'jacob':jacobBot,
        'code':codeBot,
        'prob':probBot 
     }
    }
)

