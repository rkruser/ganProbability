# testing flowgan functions
import torch
from torch.autograd import Variable
import torch.nn as nn

# Note: input must have even rows/cols
def nvpSqueeze(x):
	rows = x.size(2)
	cols = x.size(3)

	colperm = torch.cat([torch.arange(0,cols,2),torch.arange(1,cols,2)]).long()
	rowperm = torch.cat([torch.arange(0,rows,2),torch.arange(1,rows,2)]).long()
	horizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.size(0),x.size(1),x.size(2),1)
	verticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(x.size(0),x.size(1),1,x.size(3))

	if isinstance(x,Variable):
		horizontalIndex = Variable(horizontalIndex)
		verticalIndex = Variable(verticalIndex) 

	x = torch.gather(x,3,horizontalIndex) # Need to use gather function because direct slicing isn't implemented here
	x = torch.gather(x,2,verticalIndex)

	x1 = x[:,:,:rows/2,:cols/2]
	x2 = x[:,:,:rows/2,cols/2:]
	x3 = x[:,:,rows/2:,:cols/2]
	x4 = x[:,:,rows/2:,cols/2:]
	x = torch.cat([x1,x2,x3,x4],dim=1)
	return x


def nvpUnsqueeze(y):
	channels = y.size(1)

	x1 = y[:,:channels/4,:,:]
	x2 = y[:,channels/4:channels/2,:,:]
	x3 = y[:,channels/2:3*channels/4,:,:]
	x4 = y[:,3*channels/4:,:,:]	

	x12 = torch.cat([x1,x2],dim=3)
	x34 = torch.cat([x3,x4],dim=3)
	x = torch.cat([x12,x34],dim=2)

	rows = x.size(2)
	cols = x.size(3)

	rowperm = torch.zeros(rows)
	colperm = torch.zeros(cols)
	rowperm[torch.arange(0,cols,2).long()] = torch.arange(0,cols/2)
	rowperm[torch.arange(1,cols,2).long()] = cols/2+torch.arange(0,cols/2)
	colperm[torch.arange(0,rows,2).long()] = torch.arange(0,rows/2)
	colperm[torch.arange(1,rows,2).long()] = rows/2+torch.arange(0,rows/2)
	rowperm = rowperm.long()
	colperm = colperm.long()

	horizontalIndex = colperm.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.size(0),x.size(1),x.size(2),1)
	verticalIndex = rowperm.unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(x.size(0),x.size(1),1,x.size(3))

	if isinstance(y,Variable):
		horizontalIndex = Variable(horizontalIndex)
		verticalIndex = Variable(verticalIndex)

	x = torch.gather(x,2,verticalIndex)
	x = torch.gather(x,3,horizontalIndex)
	return x


def test(variable=False):
	a = torch.arange(0,72).resize_((2,6,6)).unsqueeze(0)
	if variable:
		a = Variable(a)
	b = nvpSqueeze(a)
	c = nvpUnsqueeze(b)

	print a
	print b
	print c

if __name__=='__main__':
	test(Variable)