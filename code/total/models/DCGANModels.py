from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init, NetG28, NetD28, NetG64, NetD64

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np 

# Todo:
#  Make this class slightly more general,
#  (allowing multiple colors and sizes)
#  and have things like DCGAN28 be small wrappers on top of this?

# Check for correctness
# Fill in sampling and other functions

class DCGANSize28Col3(DCGANModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 28
    args['netGclass'] = NetG28
    args['netDclass'] = NetD28
    super(DCGANSize28Col3, self).__init__(config, args)

class DCGANSize28Col1(DCGANModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 28
    args['netGclass'] = NetG28
    args['netDclass'] = NetD28
    super(DCGANSize28Col1, self).__init__(config, args)

class DCGANSize64Col1(DCGANModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 1
    args['imSize'] = 64
    args['netGclass'] = NetG64
    args['netDclass'] = NetD64
    super(DCGANSize64Col1, self).__init__(config, args)

class DCGANSize64Col3(DCGANModel):
  def __init__(self, config, args):
    args = copy(args)
    args['nc'] = 3
    args['imSize'] = 64
    args['netGclass'] = NetG64
    args['netDclass'] = NetD64
    super(DCGANSize64Col3, self).__init__(config, args)


# Need to do random seeding in another module

class DCGANModel(ModelTemplate):
  def __init__(self, config, args):
    super(DCGANModel,self).__init__(config, args)
    self.opt = {
      'nz': 100,
      'ngf':64,
      'ndf':64,
      'nc':3,
      'imSize':28,
      'ngpu':0,
      'cuda':False,
      'lr':0.0002,
      'beta1':0.5,
      'netGclass':NetG28,
      'netG':'',
      'netGinstance':-1,
      'netGexpNum':-1,
      'netDclass':NetD28,
      'netD':'',
      'netDinstance':-1,
      'netDexpNum':-1,
      'checkpointEvery':10
    }
    args = copy(args)
    self.opt.update(args)

    self.log(str(self.opt))

    # Perhaps there is a pythonic way to convert dict items to attributes
    self.nz = self.opt['nz']
    self.ngf = self.opt['ngf']
    self.ndf = self.opt['ndf']
    self.nc = self.opt['nc']
    self.imSize = self.opt['imSize']
    self.lr = self.opt['lr']
    self.beta1 = self.opt['beta1']
    self.ngpu = self.opt['ngpu']
    self.cuda = self.opt['cuda']
    self.outShape = (self.nc,self.imSize,self.imSize)

    self.log("outShape is "+str(self.outShape))

    self.netGclass = self.opt['netGclass']
    self.netGkey = self.opt['netG']
    self.netGinstance = self.opt['netGinstance']
    self.netGexpNum = self.opt['netGexpNum']

    self.netDclass = self.opt['netDclass']
    self.netDkey = self.opt['netD']
    self.netDinstance = self.opt['netDinstance']
    self.netDexpNum = self.opt['netDexpNum']

    self.checkpointEvery = self.opt['checkpointEvery']


    self.netG = self.netGclass(nz=self.nz, ngf=self.ngf, nc=self.nc, ngpu=self.ngpu)
    self.netD = self.netDclass(nz=self.nz, ngf=self.ngf, nc=self.nc, ngpu=self.ngpu)
    self.criterion = nn.BCELoss()
    self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1,0.999))
    self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1,0.999))
    self.scheduler = None

    # Load netG, newly or from file
    if self.netGkey != '':
      self.netG.load_state_dict(self.load(self.netGkey, instance=self.netGinstance, number=self.netGexpNum))
    else:
      self.netG.apply(weights_init)

    # Load netD, newly or from file
    if self.netDkey != '':
      self.netD.load_state_dict(self.load(self.netDkey, instance=self.netDinstance, number=self.netDexpNum))
    else:
      self.netD.apply(weights_init)

    if self.cuda:
      self.netG.cuda()
      self.netD.cuda()
      self.criterion.cuda()

  # data tracking
    self.images = []
    self.errG = []
    self.errD = []


  def train(self, loaderTemplate, nepochs):
    # Reset for this run
    self.images = []
    self.errG = []
    self.errD = []

    dataloader = loaderTemplate.getDataloader(outShape=self.outShape, mode='train', returnClass=False)

    for epoch in range(nepochs):
      self.log("===Begin epoch {0}".format(epoch))
      gLosses = AverageMeter()
      dLosses = AverageMeter()
      for i, data in enumerate(dataloader):
        self.log("Iteration {0}".format(i))
        
        self.netD.zero_grad()
        batchSize = data.size(0)
        labelsReal = torch.Tensor(batchSize).fill_(1.0)
        labelsFake = torch.Tensor(batchSize).fill_(0.0)
        zCodesD = torch.Tensor(batchize, self.nz, 1,1).normal_(0,1)
        zCodesG = torch.Tensor(batchize, self.nz, 1,1).normal_(0,1)

        if self.cuda:
          data.cuda()
          zCodesD.cuda()
          zCodesG.cuda()
          labelsReal.cuda()
          labelsFake.cuda()

        data = Variable(data)
        labelsReal = Variable(labelsReal)
        zCodesD = Variable(zCodesD)
        zCodesG = Variable(zCodesG)
        labelsFake = Variable(labelsFake)
        
        # Running through discriminator
        dPredictionsReal = self.netD(data)
        errDreal = self.criterion(dPredictionsReal, labelsReal)
        errDreal.backward()

        fakeImsD = self.netG(zCodesD)
        dPredictionsFake = self.netD(fakeImsD.detach())
        errDfake = self.criterion(dPredictionsFake, labelsFake)
        errDfake.backward()

        errD = errDreal + errDfake
        self.optimizerD.step()
        
        # Running through Generator
          # Do I need to sample this separately?
          # or can I use existing samples?
        self.netG.zero_grad()
        fakeImsG = self.netG(zCodesG)
        gPredictionsFake = self.netD(fakeImsG)
        errG = self.criterion(gPredictionsFake, labelsReal)
        errG.backward()
        self.optimizerG.step()

        # Extract data from run
        gLosses.update(errG.data[0], batchSize)
        dLosses.update(errD.data[0], batchSize)
        self.errG.append(gLosses.avg)
        self.errD.append(dLosses.avg)

        if i == (len(dataloader)-2) and (epoch+1)%self.checkpointEvery == 0:
          self.images.append(np.array((np.transpose(fakeImsG.data.cpu().numpy(),(0,2,3,1))[:16]*0.5+0.5)*255,dtype='uint8'))

      # Save checkpoint
      if (epoch+1)%self.checkpointEvery == 0:
        self.save(checkpointNum = epoch)

    self.save()
      
  # Sample 
  def sample(self, nSamples):
    pass

  # method should be one of 'numerical', 'exact'
  def probSample(self, nSamples, deepFeatures=None, method='numerical', epsilon=1e-5):
    codes = torch.FloatTensor(nSamples,self.nz).normal_(0,1)
    results = self.getProbs(codes, method=method, epsilon=epsilon)
    results['codes'] = codes.cpu().numpy()
    # if deepFeatures is not None:
    #   Run the deep feature network on all images and put in results
    return results

  # Move this to a library function later
  # Get the probabilities of a set of codes
  def getProbs(self, codes, method='numerical', epsilon=1e-5):
    if method=='numerical':
      noise = torch.FloatTensor(2*self.nz+1,self.nz,1,1)
    else:
      noise = torch.FloatTensor(1, self.nz, 1, 1) 
    b = torch.eye(self.nz)*epsilon # Add/subtract to code

    if self.cuda:
      noise.cuda()
#      b.cuda() # not necessary?

    gauss_const = -self.nz*np.log(np.sqrt(2*np.pi))
    log_const = 1.0

    nSamples = codes.size(0)
    images = np.empty([nSamples, self.nc, self.imSize, self.imSize])
    probs = np.empty([nSamples])
    jacob = np.empty([nSamples, self.nz])
    nX = self.nc*self.imSize*self.imSize

    self.netG.eval()
    for i in range(codes.size(0)):
      self.log("Probability sampling {0}".format(i))

      J = np.empty([nX, self.nz])

      a = codes[0].view(1,-1)
      if method=='numerical':
        noise.copy_(torch.cat((a,a+b,a-b),0).unsqueeze(2).unsqueeze(3))
        noisev = Variable(noise, volatile=True) #volatile helps with memory?
        fake = self.netG(noisev)
        I = fake.data.cpu().numpy().reshape(2*self.nz+1,-1)
        J = (I[1:nz+1,:]-I[nz+1:,:]).transpose()/(2*epsilon)
      else:
        noise.copy_(a.unsqueeze(2).unsqueeze(3))
        noisev = Variable(noise, requires_grad=True)
        fake = self.netG(noisev)
        fake = fake.view(1,-1)

        for k in range(nX):
          self.netG.zero_grad()
          fake[0,k].backward(retain_variables=True)
          J[k] = noisev.grad.data.cpu().numpy().squeeze()
        I = fake.data.cpu().numpy()

      images[i] = I[0,:].reshape(self.nc, self.imSize, self.imSize)
      
      R = np.linalg.qr(J, mode='r')
      dummy = R.diagonal().copy()
      jacob[i] = dummy.copy() # No modification yet
      dummy[np.where(np.abs(dummy) < 1e-20)] = 1
      probs[i] = -log_const*0.5*np.sum(Z**2)+gauss_const - np.log(np.abs(dummy)).sum()

    
		# What about codes?
    allData = {'images': images.astype(np.float32),
               'prob': probs.astype(np.float32), #no s to be consistent
               'jacob': jacob.astype(np.float32)}
    return allData


      

  # Get the generator codes that reconstruct the images most closely
  def nearestInput(self, ims):
    pass

  def test(self, loaderTemplate, nepochs):
    pass



  def saveCheckpoint(self, checkpointNum=None):
    if checkpointNum is not None:
      self.save(self.netG.state_dict(), 'netG', instance=checkpointNum, saver='torch')
      self.save(self.netD.state_dict(), 'netD', instance=checkpointNum, saver='torch')
    else:
      self.save(self.netG.state_dict(), 'netG', saver='torch')
      self.save(self.netD.state_dict(), 'netD', saver='torch')


  def getAnalysisData(self):
    assert(len(self.errG) == len(self.errD))
    errInfo = {
      'data':np.array([range(len(self.errG)),self.errG,self.errD]),
      'legend':['Generator error','Discriminator Error'],
      'xlabel':'Epoch',
      'ylabel':'Average error',
      'title':'GAN training curve',
      'format':'png'
    }
    errAnalysis = Data(errInfo, 'lineplot', 'ganTrainPlot')
    results = [errAnalysis]
    for i, im in enumerate(self.images):
      results.append(Data({'images':im,'dpi':400}, 'imageArray', 'ganSampleIms', instance=(i*self.checkpointEvery)))
    return results

