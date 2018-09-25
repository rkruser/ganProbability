from copy import copy
from mlworkflow import Operator

class StandardExperiment(Operator):
	def __init__(self, config, args):
		super(StandardExperiment, self).__init__(config, args)
		args = copy(args)
		self.opt = {
			'nEpochs':20
		}
		self.opt.update(args)

		self.loaderTemplate = self.dependencies[0]
		self.ganModel = self.dependencies[1]
		self.regressorModel = self.dependencies[2]
		self.sampler = self.dependencies[3]


	def run(self):
		self.ganModel.train(self.loaderTemplate, self.opt['nEpochs'])




