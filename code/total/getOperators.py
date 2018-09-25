import sys
from code.total.loaders.MatLoaders import MNISTSize28Cols1, MNISTSize32Cols1, MNISTSize64Cols1, CFARSize28Cols3, CFARSize32Cols3, CFARSize64Cols3, CUB200Size64Cols3
from code.total.models.DCGANModels import DCGANSize28Col3, DCGANSize28Col1, DCGANSize32Col3, DCGANSize32Col1, DCGANSize64Col1, DCGANSize64Col3
from code.total.models.RegressorModels import RegressorSize28Col3, RegressorSize28Col1, RegressorSize32Col3, RegressorSize32Col1, RegressorSize64Col3, RegressorSize64Col1

def getModules(mlist):
  mods = []
  for m in mlist:
    toAppend = None
    # Dataloaders
    if m == 'MNISTSize28Cols1':
      toAppend = MNISTSize28Cols1
    elif m == 'MNISTSize32Cols1':
      toAppend = MNISTSize32Cols1
    elif m == 'MNISTSize64Cols1':
      toAppend = MNISTSize64Cols1
    elif m == 'CFARSize28Cols3':
      toAppend = CFARSize28Cols3
    elif m == 'CFARSize32Cols3':
      toAppend = CFARSize32Cols3
    elif m == 'CFARSize64Cols3':
      toAppend = CFARSize64Cols3
    elif m == 'CUB200Size64Cols3':
      toAppend = CUB200Size64Cols3
    # Models
    elif m == 'DCGANSize28Col3':
      toAppend = DCGANSize28Col3
    elif m == 'DCGANSize28Col1':
      toAppend = DCGANSize28Col1
    elif m == 'DCGANSize32Col3':
      toAppend = DCGANSize32Col3
    elif m == 'DCGANSize32Col1':
      toAppend = DCGANSize32Col1
    elif m == 'DCGANSize64Col1':
      toAppend = DCGANSize64Col1
    elif m == 'DCGANSize64Col3':
      toAppend = DCGANSize64Col3
    # Regressors
    elif m == 'RegressorSize28Col3':
      toAppend = RegressorSize28Col3
    elif m == 'RegressorSize28Col1':
      toAppend = RegressorSize28Col1
    elif m == 'RegressorSize32Col3':
      toAppend = RegressorSize32Col3
    elif m == 'RegressorSize32Col1':
      toAppend = RegressorSize32Col1
    elif m == 'RegressorSize64Col3':
      toAppend = RegressorSize64Col3
    elif m == 'RegressorSize64Col1':
      toAppend = RegressorSize64Col1
    # Experiments
    
    if toAppend is None:
      print "No module named {0}".format(m)
      sys.exit(1)
    mods.append(toAppend)

  return mods


    
    
    
