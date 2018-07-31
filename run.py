# Run experiments
# Import modules from code here
from mlworkflow import Loader, Operator, Analyzer, experiment
from mpi4py import MPI

from code.LoadGAN import LoadGAN
from code.TrainGAN import TrainGAN

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id',type=int,default=-1,help='experiment ID')
parser.add_argument('--procID',type=int,default=-1,help='Batch ID')
parser.add_argument('--nprocs',type=int,default=-1,help='Number of batch processes')
parser.add_argument('--masterconfig',type=str,default='/fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml',
    help='Path to master config')


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
  return classes

def main():
  opt = parser.parse_args()
  comm = MPI.COMM_WORLD

  rank = opt.procID
  if rank < 0:
    rank = comm.Get_rank()

  size = opt.nprocs
  if size < 0:
    size = comm.Get_size()

#  s = getClasses(['LoadGAN','TrainGAN'])
#  print s

  experiment(masterconfig=opt.masterconfig, classParser=getClasses, experimentNum=opt.id, pid=rank, numProcs=size,verbose=True)

if __name__ == '__main__':
  main()

