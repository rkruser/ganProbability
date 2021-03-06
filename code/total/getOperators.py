from code.total.loaders.MatLoaders import MNISTSize28Cols1, MNISTSize32Cols1, MNISTSize64Cols1, CFAR10Size28Cols3, CFAR10Size32Cols3, CFAR10Size64Cols3, CUB200Size64Cols3, CUB200Size32Cols3, BirdsnapSize64Cols3, BirdsnapSize32Cols3, ProbData
from code.total.models.DCGANModels import DCGANSize28Col3, DCGANSize28Col1, DCGANSize32Col3, DCGANSize32Col1, DCGANSize64Col1, DCGANSize64Col3
from code.total.models.RegressorModels import RegressorSize28Col3, RegressorSize28Col1, RegressorSize32Col3, RegressorSize32Col1, RegressorSize64Col3, RegressorSize64Col1
from code.total.models.RegressorModels import DeepRegressorSize28Col3, DeepRegressorSize28Col1, DeepRegressorSize32Col3, DeepRegressorSize32Col1, DeepRegressorSize64Col3, DeepRegressorSize64Col1, FeatRegressor10, FeatRegressor500
from code.total.models.flowGAN import FlowGANModel
from code.total.models.realNVP import RealNVP
from code.total.models.DeepFeatures import LenetSize28Cols3, LenetSize28Cols1, LenetSize32Cols3, LenetSize32Cols1, LenetSize64Cols3, LenetSize64Cols1, LenetSize128Cols3, LenetSize128Cols1
from code.total.experiments.Experiments import GANTrain, RegressorTrain, RegressorTest, DeepFeatureTrain
from code.total.experiments.SampleGAN import SampleGAN, SampleDeepGAN
from code.total.experiments.ZipSamples import ZipSamples

mapDict = {
   # Loaders
    'MNISTSize28Cols1': MNISTSize28Cols1,
    'MNISTSize32Cols1': MNISTSize32Cols1, 
    'MNISTSize64Cols1': MNISTSize64Cols1,
    'CFAR10Size28Cols3': CFAR10Size28Cols3,
    'CFAR10Size32Cols3': CFAR10Size32Cols3,
    'CFAR10Size64Cols3': CFAR10Size64Cols3,
    'CUB200Size64Cols3': CUB200Size64Cols3,
    'CUB200Size32Cols3': CUB200Size32Cols3,
    'BirdsnapSize64Cols3': BirdsnapSize64Cols3,
    'BirdsnapSize32Cols3': BirdsnapSize32Cols3,
    'ProbData': ProbData,
   # Models 
    'DCGANSize28Col3': DCGANSize28Col3,
    'DCGANSize28Col1': DCGANSize28Col1,
    'DCGANSize32Col3': DCGANSize32Col3,
    'DCGANSize32Col1': DCGANSize32Col1,
    'DCGANSize64Col1': DCGANSize64Col1,
    'DCGANSize64Col3': DCGANSize64Col3,
    'FlowGAN': FlowGANModel,
   # Regressors 
    'RegressorSize28Col3': RegressorSize28Col3,
    'RegressorSize28Col1': RegressorSize28Col1,
    'RegressorSize32Col3': RegressorSize32Col3,
    'RegressorSize32Col1': RegressorSize32Col1,
    'RegressorSize64Col3': RegressorSize64Col3,
    'RegressorSize64Col1': RegressorSize64Col1,
   # Deep Regressors 
    'DeepRegressorSize28Col3': DeepRegressorSize28Col3,
    'DeepRegressorSize28Col1': DeepRegressorSize28Col1,
    'DeepRegressorSize32Col3': DeepRegressorSize32Col3,
    'DeepRegressorSize32Col1': DeepRegressorSize32Col1,
    'DeepRegressorSize64Col3': DeepRegressorSize64Col3,
    'DeepRegressorSize64Col1': DeepRegressorSize64Col1,
    'FeatRegressor10': FeatRegressor10, #Really the only true feature regressor
    'FeatRegressor500': FeatRegressor500,
   # Deep features
    'LenetSize28Cols1': LenetSize28Cols1,
    'LenetSize28Cols3': LenetSize28Cols3,
    'LenetSize32Cols1': LenetSize32Cols1,
    'LenetSize32Cols3': LenetSize32Cols3,
    'LenetSize64Cols1': LenetSize64Cols1,
    'LenetSize64Cols3': LenetSize64Cols3,
    'LenetSize128Cols1': LenetSize128Cols1,
    'LenetSize128Cols3': LenetSize128Cols3,
   # Sampling
    'SampleGAN': SampleGAN,
    'SampleDeepGAN': SampleDeepGAN,
    'ZipSamples': ZipSamples,
   # Experiments
    'GANTrain': GANTrain,
    'RegressorTrain': RegressorTrain,
    'RegressorTest': RegressorTest,
    'DeepFeatureTrain': DeepFeatureTrain
}


