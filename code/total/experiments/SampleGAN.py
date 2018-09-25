from copy import copy
from mlworkflow import Operator

from scipy.io import savemat


class SampleGAN(Operator):
	def __init__(self, config, args):
		super(SampleGAN, self).__init__(config, args)
		args = copy(args)
		self.opt = {
			'samples':10000,
			'deep':False,
			'method':'numerical',
			'epsilon':1e-5,
			'appendThreadId':True,
			'autoRun':True
		}
		self.opt.update(args)
		self.ganModel = self.dependencies[0]

	def sample(self):
		samples = self.ganModel.probSample(nSamples = self.opt['samples'], deepFeatures = self.opt['deep'],
				method=self.opt['method'], epsilon=self.opt['epsilon'])
		self.save(samples, 'samples', saver='mat', threadSpecific = self.opt['appendThreadId'])

	def run(self):
    # Check if netG file exists
		if self.opt['autoRun']:
			self.sample()



