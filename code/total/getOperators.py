import sys
from code.total.loaders.MatLoaders import MNISTSize28Cols1, MNISTSize32Cols1, MNISTSize64Cols1, CFARSize28Cols3, CFARSize32Cols3, CFARSize64Cols3, CUB200Size64Cols3
from code.total.models.DCGANModels import DCGANSize28Col3, DCGANSize28Col1, DCGANSize32Col3, DCGANSize32Col1, DCGANSize64Col1, DCGANSize64Col3
from code.total.models.RegressorModels import RegressorSize28Col3, RegressorSize28Col1, RegressorSize32Col3, RegressorSize32Col1, RegressorSize64Col3, RegressorSize64Col1

mapDict = {
   # Loaders
    'MNISTSize28Cols1': MNISTSize28Cols1,
    'MNISTSize32Cols1': MNISTSize32Cols1, 
    'MNISTSize64Cols1': MNISTSize64Cols1,
    'CFARSize28Cols3': CFARSize28Cols3,
    'CFARSize32Cols3': CFARSize32Cols3,
    'CFARSize64Cols3': CFARSize64Cols3,
    'CUB200Size64Cols3': CUB200Size64Cols3,
   # Models 
    'DCGANSize28Col3': DCGANSize28Col3,
    'DCGANSize28Col1': DCGANSize28Col1,
    'DCGANSize32Col3': DCGANSize32Col3,
    'DCGANSize32Col1': DCGANSize32Col1,
    'DCGANSize64Col1': DCGANSize64Col1,
    'DCGANSize64Col3': DCGANSize64Col3,
   # Regressors 
    'RegressorSize28Col3': RegressorSize28Col3,
    'RegressorSize28Col1': RegressorSize28Col1,
    'RegressorSize32Col3': RegressorSize32Col3,
    'RegressorSize32Col1': RegressorSize32Col1,
    'RegressorSize64Col3': RegressorSize64Col3,
    'RegressorSize64Col1': RegressorSize64Col1
   # Experiments
}



    
    
    
