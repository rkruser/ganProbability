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

mapDict = {
    'Operator': Operator,
    'Loader': Loader,
    'Analyzer': Analyzer,
    'LoadGAN': LoadGAN,
    'TrainGAN': TrainGAN,
    'SampleGAN': SampleGAN,
    'ZipSamples': ZipSamples,
    'RegressorTraining': RegressorTraining,
    'RegressorRun': RegressorRun,
    'DataloaderRyen': DataloaderRyen,
    'ModelLoaderRyen': ModelLoaderRyen,
    'TrainerRyen': TrainerRyen,
    'SampleAnalyze': SampleAnalyze,
    'RegressorTrainingRyen': RegressorTrainingRyen,
    'RegressorRun3Col': RegressorRun3Col
}

