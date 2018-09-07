# Run experiments
# Import modules from code here
from mlworkflow import Loader, Operator, Analyzer, experiment
from mpi4py import MPI

#import code.sohil as sohil

#LoadGAN = TrainGAN = SampleGAN = ZipSamples = RegressorTraining = RegressorRun = None

from code.sohil.LoadGAN import LoadGAN
from code.sohil.TrainGAN import TrainGAN
from code.sohil.SampleGAN import SampleGAN
from code.sohil.MergeSamples import ZipSamples
from code.sohil.RegressorTraining import RegressorTraining
from code.sohil.RegressorRunning import RegressorRun
from code.sohil.RegressorRunning3Col import RegressorRun3Col

from code.ryen.loaders import DataloaderRyen
from code.ryen.models import ModelLoaderRyen
from code.ryen.trainer import TrainerRyen
from code.ryen.sampling import SampleAnalyze
from code.ryen.regressorTraining import RegressorTrainingRyen

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id',type=int,default=-1,help='experiment ID')
parser.add_argument('--pid',type=int,default=-1,help='Batch ID')
parser.add_argument('--nprocs',type=int,default=-1,help='Number of batch processes')
# Need to make the default './master.yaml'
parser.add_argument('--masterconfig',type=str,default='./master.yaml',
    help='Path to master config')
parser.add_argument('--stage', type=int, default=1, help='The stage of the model to run')


def getClasses(clist):
  classes = []
  for c in clist:
    if c == 'Operator':
      classes.append(Operator)
    elif c == 'Loader':
      classes.append(Loader)
    elif c == 'Analyzer':
      classes.append(Analyzer)
    elif c == 'LoadGAN':
      classes.append(LoadGAN)
    elif c == 'TrainGAN':
      classes.append(TrainGAN)
    elif c == 'SampleGAN':
      classes.append(SampleGAN)
    elif c == 'ZipSamples':
      classes.append(ZipSamples)
    elif c == 'RegressorTraining':
      classes.append(RegressorTraining)
    elif c == 'RegressorRun':
      classes.append(RegressorRun)
    elif c == 'DataloaderRyen':
      classes.append(DataloaderRyen)
    elif c == 'ModelLoaderRyen':
      classes.append(ModelLoaderRyen)
    elif c == 'TrainerRyen':
      classes.append(TrainerRyen)
    elif c == 'SampleAnalyze':
      classes.append(SampleAnalyze)
    elif c == 'RegressorTrainingRyen':
      classes.append(RegressorTrainingRyen)
    elif c == 'RegressorRun3Col':
      classes.append(RegressorRun3Col)
  return classes

def main():
  opt = parser.parse_args()
  comm = MPI.COMM_WORLD

  rank = opt.pid
  if rank < 0:
    rank = comm.Get_rank()

  size = opt.nprocs
  if size < 0:
    size = comm.Get_size()

#  s = getClasses(['LoadGAN','TrainGAN'])
#  print s

  experiment(masterconfig=opt.masterconfig, classParser=getClasses, experimentNum=opt.id, pid=rank, numProcs=size,verbose=True,stage=opt.stage)

if __name__ == '__main__':
  main()

