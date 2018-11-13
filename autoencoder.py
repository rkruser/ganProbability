import torch
import torch.nn as nn

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.main = nn.Sequential(
				nn.Linear(3072, 1024),
				nn.ReLU(inplace=True),
				nn.Linear(1024, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256)
			)

	def numLatent(self):
		return None

	def numOutDims(self):
		return 256

	def outshape(self):
		return [256]

	def imsize(self):
		return [3,32,32]

	def numColors(self):
		return None


	def forward(self, x):
		out = x.view(x.size(0),-1)
		out = self.main(out)
		return out

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.main = nn.Sequential(
			nn.Linear(256, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512,1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024,3072),
			nn.Tanh()
			)

	def numLatent(self):
		return None

	def numOutDims(self):
		return 3072

	def outshape(self):
		return [3,32,32]

	def imsize(self):
		return None

	def numColors(self):
		return None


	def forward(self, x):
		out = self.main(x)
		out = out.view(x.size(0), 3, 32, 32)
		return out


class Encoder2(nn.Module):
	def __init__(self):
		super(Encoder2, self).__init__()
		self.main = nn.Sequential(
				nn.Linear(3072, 2048),
				nn.ReLU(inplace=True),
				nn.Linear(2048,1024),
				nn.ReLU(inplace=True),
				nn.Linear(1024,784),
				nn.ReLU(inplace=True),
				nn.Linear(784,512),
				nn.ReLU(inplace=True),
				nn.Linear(512,364),
				nn.ReLU(inplace=True),
				nn.Linear(364,256)
			)

	def numLatent(self):
		return None

	def numOutDims(self):
		return 256

	def outshape(self):
		return [256]

	def imsize(self):
		return [3,32,32]

	def numColors(self):
		return None


	def forward(self, x):
		out = x.view(x.size(0),-1)
		out = self.main(out)
		return out

class Decoder2(nn.Module):
	def __init__(self):
		super(Decoder2,self).__init__()
		self.main = nn.Sequential(
				nn.Linear(256, 364),
				nn.ReLU(inplace=True),
				nn.Linear(364, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 784),
				nn.ReLU(inplace=True),
				nn.Linear(784, 1024),
				nn.ReLU(inplace=True),
				nn.Linear(1024, 2048),
				nn.ReLU(inplace=True),
				nn.Linear(2048, 3072),
				nn.Tanh()
			)

	def numLatent(self):
		return None

	def numOutDims(self):
		return 3072

	def outshape(self):
		return [3,32,32]

	def imsize(self):
		return None

	def numColors(self):
		return None


	def forward(self, x):
		out = self.main(x)
		out = out.view(x.size(0), 3, 32, 32)
		return out
