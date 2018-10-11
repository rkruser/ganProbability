from copy import copy
from mlworkflow import Operator

from scipy.io import savemat

from code.total.models.nnModels import NthArgWrapper


class SampleGAN(Operator):
	def __init__(self, config, args):
		super(SampleGAN, self).__init__(config, args)
		args = copy(args)
		self.opt = {
      'sampleKey':'samples',
			'samples':10000,
   		'resample':False,
			'deep':False,
			'method':'numerical',
			'epsilon':1e-5,
			'appendThreadId':True,
			'autoRun':True
		}
		self.opt.update(args)
		self.ganModel = self.dependencies[0]

	def sample(self):
		if self.opt['resample'] or (not self.checkExists(self.opt['sampleKey'], threadSpecific = self.opt['appendThreadId'])):
			samples = self.ganModel.probSample(nSamples = self.opt['samples'], deepFeatures = None, method=self.opt['method'], epsilon=self.opt['epsilon'])
			self.save(samples, self.opt['sampleKey'], saver='mat', threadSpecific = self.opt['appendThreadId'])

	def run(self):
    # Check if netG file exists
		if self.opt['autoRun']:
			self.sample()


class SampleDeepGAN(Operator):
	def __init__(self, config, args):
		super(SampleDeepGAN, self).__init__(config, args)
		args = copy(args)
		self.opt = {
      		'sampleKey':'samples',
			'samples':10000,
      		'resample':False,
			'deep':False,
			'featsOut':10,
			'method':'numerical',
			'epsilon':1e-5,
			'appendThreadId':True,
			'autoRun':True
		}
		self.opt.update(args)
		self.ganModel = self.dependencies[0]
		self.deepModel = self.dependencies[1]

	def sample(self):
		if self.opt['resample'] or (not self.checkExists(self.opt['sampleKey'], threadSpecific = self.opt['appendThreadId'])):
			if self.opt['featsOut'] == 10:
				argOut = 1
			else:
				argOut = 0
			embeddingNet = NthArgWrapper(self.deepModel.getModel(), argOut)
			samples = self.ganModel.probSample(nSamples = self.opt['samples'], deepFeatures = embeddingNet,
			 			deepFeaturesOutsize=self.opt['featsOut'], method=self.opt['method'], epsilon=self.opt['epsilon'])
			self.save(samples, self.opt['sampleKey'], instance=self.getPID(), saver='mat')#, threadSpecific = self.opt['appendThreadId'])

	def run(self):
    # Check if netG file exists
		if self.opt['autoRun']:
			self.sample()

