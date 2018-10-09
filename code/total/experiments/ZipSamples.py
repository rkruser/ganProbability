# Extract the top and bottom so many images
# Transform to viewable format

from mlworkflow import Operator, Data
from easydict import EasyDict as edict

import scipy.io as sio #use sio.loadmat and sio.savemat
import numpy as np
import argparse
import os.path as osp
import sys

class ZipSamples(Operator):
  def __init__(self, config, args):
    super(ZipSamples,self).__init__(config, args)
    opt = {
      'zipkey':'samples',
      'collapseInterval':1,
      'collapseTimeout':300,
      'take':25
    }
    opt.update(args)
    self.opt = edict(opt)
    self.analysisData = []

  def run(self):
    # Collapse into one process
    nprocs = self.getNumProcs()
    self.log("Collapsing to process 0")
    self.collapse(pid=0,timeout=self.opt.collapseTimeout,interval=self.opt.collapseInterval)
    self.log("In Process 0 (this should not be seen by other processes)")

    # Do nothing if already have sampled
    if self.checkExists(self.opt.zipkey, threadSpecific=False):
      return

    opt = self.opt

    self.log("Loading mats")
    loadedMats = []
    for i in range(nprocs):
      loadedMats.append(self.files.load(self.opt.zipkey, instance=i, loader='mat'))

#    keys = [k for k in loadedMats[0]]

#    allSamples = {k: np.concatenate([loadedMats[i][k] for i in range(nprocs)]) for k in keys if not k.startswith('__')}

    allSamples = {
      'images': np.concatenate([loadedMats[i]['images'] for i in range(nprocs)]),
      'jacob': np.concatenate([loadedMats[i]['jacob'] for i in range(nprocs)]),
      'code': np.concatenate([loadedMats[i]['code'] for i in range(nprocs)]),
      'prob': np.concatenate([loadedMats[i]['prob'].flatten() for i in range(nprocs)])
    }

    if 'feats' in loadedMats[0]:
      allSamples['feats'] = np.concatenate([loadedMats[i]['feats'] for i in range(nprocs)])

    self.log("Saving %s"%self.opt.zipkey)
    self.files.save(allSamples,self.opt.zipkey,saver='mat',threadSpecific=False)

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
  #  imTop = (imTop+1.0)*127.5
  #  imTop = imTop.astype(np.uint8)
    imTop = imTop*0.5+0.5 #Transform changes depending on dataset
    # Later make the transform flexible
#    imTop = np.transpose(imTop,(2,3,1,0)) # Transpose to RGB format
    imTop = np.transpose(imTop,(0,2,3,1))
    # put number of samples in last dimension because
    # that's hows matlab needs it for imshow

    self.analysisData.append(Data({'images':imTop},'imageArray','topIms'))

    imBot = allSamples['images'][bottomN]
    probBot = allSamples['prob'][bottomN]
    jacobBot = allSamples['jacob'][bottomN]
    codeBot = allSamples['code'][bottomN]
#    imBot = (imBot+1.0)*127.5
#    imBot = imBot.astype(np.uint8)
#    imBot = np.transpose(imBot,(2,3,1,0))
    imBot = imBot*0.5+0.5
    imBot = np.transpose(imBot,(0,2,3,1))

    self.analysisData.append(Data({'images':imBot},'imageArray','botIms')) 

#    sio.savemat(osp.join(opt.matfolder,'topBottom%d.mat'%opt.take), 
#        {
#          'top':{
#            'images':imTop,
#            'jacob':jacobTop,
#            'code':codeTop,
#            'prob':probTop 
#          },
#         'bottom':{
#            'images':imBot,
#            'jacob':jacobBot,
#            'code':codeBot,
#            'prob':probBot 
#         }
#        }
#    )
  def getAnalysisData(self):
    return self.analysisData

