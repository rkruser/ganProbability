# Run experiments
# Import modules from code here
from mlworkflow import Loader, Operator, Analyzer, experiment
from mpi4py import MPI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id',type=int,default=-1,help='experiment ID')
parser.add_argument('--pid',type=int,default=-1,help='Batch ID')
parser.add_argument('--nprocs',type=int,default=-1,help='Number of batch processes')
# Need to make the default './master.yaml'
parser.add_argument('--masterconfig',type=str,default='./master.yaml',
    help='Path to master config')
parser.add_argument('--stage', type=int, default=1, help='The stage of the model to run')


def getModulesFromDict(mlist, moduleMap):
  mods = []
  for m in mlist:
    toAppend = moduleMap.get(m)
    if toAppend is None:
      print "No module named {0}".format(m)
      sys.exit(1)
    mods.append(toAppend)
  return mods

from getOperators import mapDict
#from code.total.getOperators import mapDict
getClasses = lambda mlist: getModulesFromDict(mlist, mapDict)

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

