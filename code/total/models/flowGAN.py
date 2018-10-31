from mlworkflow import Data
from copy import copy
from code.total.models.ModelTemplate import ModelTemplate, AverageMeter
from code.total.models.nnModels import weights_init, NetG32, NetD32
from code.total.models.realNVP import RealNVP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np 

# sdlfkjasdfasldfkj

# Todo:
#  Make this class slightly more general,
#  (allowing multiple colors and sizes)
#  and have things like DCGAN28 be small wrappers on top of this?

# Check for correctness
# Fill in sampling and other functions

def weights_clip(m, c=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.clamp(-c, c)
        if m.bias is not None:
          m.bias.clamp(-c,c)
    elif classname.find('BatchNorm') != -1:
        m.weight.clamp(-c, c)
        m.bias.clamp(-c,c)
    elif classname.find('Linear') != -1:
        m.weight.clamp(-c,c)
        if m.bias is not None:
          m.bias.clamp(-c,c)


# Need to do random seeding in another module

class FlowGANModel(ModelTemplate):
  def __init__(self, config, args):
    super(FlowGANModel,self).__init__(config, args)
    self.opt = {
      'nh':32,
      'ndf':64,
      'nc':3,
      'ks':3, #kernel size
      'imSize':32,
      'ngpu':0,
      'cuda':False,
      'lr':0.0002,
      'beta1':0.5,
      'lossLambda':0.1,
      'batchsize':64,
      'netGclass':RealNVP,
      'netGkey':'netG',
      'netGinstance':-1,
      'netGexpNum':-1,
      'netDclass':NetD32,
      'netDkey':'netD',
      'netDinstance':-1,
      'netDexpNum':-1,
      'checkpointEvery':5
    }
    args = copy(args)
    self.opt.update(args)

    for key in self.opt:
      setattr(self, key, self.opt[key])

    self.nz = self.nc*self.imSize*self.imSize #Same number of inputs as outputs for invertible model

    self.log(str(self.opt))


    self.outShape = (self.nc, self.imSize, self.imSize)
    self.log("outShape is "+str(self.outShape))


    self.netG = self.netGclass(self.imSize, self.nc, nh=self.nh, ks=self.ks, batchsize=self.batchsize, ngpu=self.ngpu)
    self.netD = self.netDclass(nz=self.nz, ndf=self.ndf, nc=self.nc, ngpu=self.ngpu)

    # Replace criterion with new function
    # Also, add weight clipping somewhere
    self.criterion = nn.BCELoss() 
    self.optimizerG = optim.Adam(filter(lambda m: m.requires_grad, self.netG.parameters()), lr=self.lr, betas=(self.beta1,0.999))
    self.optimizerD = optim.Adam(filter(lambda m: m.requires_grad, self.netD.parameters()), lr=self.lr, betas=(self.beta1,0.999))
    self.scheduler = None


    if (self.netGinstance == -1 or self.netDinstance == -1):
      self.netGinstance = self.getLatestInstance(self.netGkey, self.netGexpNum)
      self.netDinstance = self.getLatestInstance(self.netDkey, self.netDexpNum)

    if (self.netGinstance is not None) and (self.netGinstance == self.netDinstance):
      self.log("Loading GAN from instance {0}".format(self.netGinstance))
      self.netG.load_state_dict(self.load(self.netGkey, instance=self.netGinstance, number=self.netGexpNum, loader='torch'))
      self.netD.load_state_dict(self.load(self.netDkey, instance=self.netDinstance, number=self.netDexpNum, loader='torch'))
      if self.checkExists('ganState', instance=self.netGinstance, number=self.netGexpNum):
      # data tracking
        self.images, self.errG, self.errD = self.load('ganState', instance=self.netGinstance, number=self.netGexpNum, loader='pickle')
      else:
        self.images = []
        self.errG = []
        self.errD = []
    else:
      self.log("Starting GAN from scratch")
      self.netGinstance = -1
      self.netDinstance = -1
      self.netG.apply(weights_init)
      self.netD.apply(weights_init)
      self.images = []
      self.errG = []
      self.errD = []



    if self.cuda:
      self.netG = self.netG.cuda()
      self.netD = self.netD.cuda()
      # self.netG = nn.DataParallel(self.netG).cuda()
      # self.netD = nn.DataParallel(self.netD).cuda()
      self.criterion = self.criterion.cuda()



  def train(self, loaderTemplate, nepochs):
    # Reset for this run
    # self.images = []
    # self.errG = []
    # self.errD = []

    # Remember to keep the batch size fixed here, maybe?
    dataloader = loaderTemplate.getDataloader(outShape=self.outShape, mode='train', returnLabel=False, fuzzy=True, drop_last=True)

    gauss_const = -self.nz*np.log(np.sqrt(2*np.pi))
    log_const = 1


    self.netG.train()
    self.netD.train()
    startEpoch = self.netGinstance+1
    for epoch in range(startEpoch,nepochs):
      self.log("===Begin epoch {0}".format(epoch))
      gLosses = AverageMeter()
      dLosses = AverageMeter()
      for i, data in enumerate(dataloader):
        # if i > 10 and i<927:
        #   continue
#        if i%10 == 0:
        self.log("Iteration {0}".format(i))
        
        self.netD.zero_grad()
        batchSize = data.size(0)
        labelsReal = torch.Tensor(batchSize).fill_(1.0)
        labelsFake = torch.Tensor(batchSize).fill_(0.0)
        zCodesD = torch.Tensor(batchSize, self.nz, 1,1).normal_(0,1)
#        zCodesG = torch.Tensor(batchSize, self.nz, 1,1).normal_(0,1)

        if self.cuda:
          data = data.cuda()
          zCodesD = zCodesD.cuda()
#          zCodesG = zCodesG.cuda()
          labelsReal = labelsReal.cuda()
          labelsFake = labelsFake.cuda()

        data = Variable(data)
        labelsReal = Variable(labelsReal)
        zCodesD = Variable(zCodesD)
#        zCodesG = Variable(zCodesG)
        labelsFake = Variable(labelsFake)
        
        # Running through discriminator
        dPredictionsReal = self.netD(data)
#        errDreal = self.criterion(dPredictionsReal, labelsReal)
        errDreal = -dPredictionsReal.mean()

        fakeImsD, _ = self.netG(zCodesD, invert=True)
        dPredictionsFake = self.netD(fakeImsD.detach())
#        errDfake = self.criterion(dPredictionsFake, labelsFake)
        errDfake = dPredictionsFake.mean()

        errD = errDreal + errDfake
        errD.backward()
        self.optimizerD.step()
        weights_clip(self.netD)
        
        # Running through Generator
          # Do I need to sample this separately?
          # or can I use existing samples?
        self.netG.zero_grad()
#        fakeImsG, _ = self.netG(zCodesG, invert=True) #????
        fakeImsG = fakeImsD
        gPredictionsFake = self.netD(fakeImsG)
        dataInverted, logDetDataInverted = self.netG(data)
#        logLikelihoodLoss = gauss_const - log_const*0.5*(dataInverted**2).sum(dim=1) + logDetDataInverted
#        logLikelihoodLoss = logLikelihoodLoss.mean()
        logLikelihoodLoss = 0

#        errG = self.criterion(gPredictionsFake, labelsReal)
        errG = -gPredictionsFake.mean()
        errG = errG - self.lossLambda*logLikelihoodLoss
        errG.backward()
        self.optimizerG.step()

        # Need to weight-clip somewhere after stepping each function, if using wasserstein

        # Extract data from run
        gLosses.update(errG.data[0], batchSize)
        dLosses.update(errD.data[0], batchSize)
        self.errG.append(gLosses.avg)
        self.errD.append(dLosses.avg)
#        if i%10 == 0:
        self.log("GLoss: {0}, DLoss: {1}".format(gLosses.avg, dLosses.avg))

        if i == (len(dataloader)-2) and (epoch+1)%self.checkpointEvery == 0:
          self.images.append(np.array((np.transpose(fakeImsG.data.cpu().numpy(),(0,2,3,1))[:16]*0.5+0.5)*255,dtype='uint8'))

      # Save checkpoint
      if (epoch+1)%self.checkpointEvery == 0:
        self.saveCheckpoint(checkpointNum = epoch)

    self.saveCheckpoint(checkpointNum=(nepochs-1))
      
  # Sample 
  def sample(self, nSamples):
    self.netG.eval()
    codes = torch.FloatTensor(nSamples,self.nz).normal_(0,1)
    results, _ = self.netG.invert(codes)
    return results


  # method should be one of 'numerical', 'exact'
  # deepFeatures should be an NN object with a forward function
  def probSample(self, nSamples, deepFeatures=None, deepFeaturesOutsize=None, method='numerical', epsilon=1e-5):
    codes = torch.FloatTensor(nSamples,self.nz).normal_(0,1)
    if self.cuda:
      codes = codes.cuda()
      if deepFeatures is not None:
        deepFeatures = deepFeatures.cuda()
    results = self.getProbs(codes, deepFeatures=deepFeatures, deepFeaturesOutsize=deepFeaturesOutsize, method=method, epsilon=epsilon)
    results['code'] = codes.cpu().numpy()
    # if deepFeatures is not None:
    #   Run the deep feature network on all images and put in results
    return results

  # Move this to a library function later
  # Get the probabilities of a set of codes
  def getProbs(self, codes, deepFeatures=None, deepFeaturesOutsize=None, method='numerical', epsilon=1e-5):
    self.netG.eval()
    # Using log10 because it's more intuitive
    gauss_const = -self.nz*np.log10(np.sqrt(2*np.pi))
    log_const = np.log10(np.exp(1))
    log_div = np.log(10.0)

    nSamples = codes.size(0)
    images = np.empty([nSamples, self.nc, self.imSize, self.imSize])
    probs = np.empty([nSamples])

    if deepFeatures is None:
      jacob = None
    else:
      jacob = np.empty([nSamples, min(nX,self.nz)]) 

    if deepFeaturesOutsize is not None:
      nX = deepFeaturesOutsize
      feats = np.empty([nSamples, nX])
    else:
      nX = self.nc*self.imSize*self.imSize


    if deepFeatures is None:
      for i in range(codes.size(0)):
        z = Variable(codes[i].view(1,-1))
        im, logDetIm = self.netG.invert(Variable(codes[i].view(1,-1)))
        imProb = gauss_const-log_const*0.5*(z**2).sum()-logDetIm/log_div #change log to log 10 #subtract to get log of inverse
        probs[i] = imProb
        images[i] = im.data.cpu().numpy().reshape(self.nc, self.imSize, self.imSize)
    else:
      if method=='numerical':
        noise = torch.FloatTensor(2*self.nz+1,self.nz,1,1)
      else:
        noise = torch.FloatTensor(1, self.nz, 1, 1) 
      b = torch.eye(self.nz)*epsilon # Add/subtract to code

      if self.cuda:
        noise = noise.cuda()
        b = b.cuda()

      for i in range(codes.size(0)):
        self.log("Probability sampling {0}".format(i))

        J = np.empty([nX, self.nz])

        a = codes[i].view(1,-1)
        # Numerical method is nearly untenable here
        # Better use backprop?
        # Also, only use backprop through the deep feature layer
        if method=='numerical':
          noise.copy_(torch.cat((a,a+b,a-b),0).unsqueeze(2).unsqueeze(3))
          noisev = Variable(noise, volatile=True) #volatile helps with memory?
          fakeIms, _ = self.netG(noisev)
          if deepFeatures is not None:
            fake = deepFeatures(fakeIms) # What if there are multiple return values
          else:
            fake = fakeIms
          I = fake.data.cpu().numpy().reshape(2*self.nz+1,-1)
          J = (I[1:self.nz+1,:]-I[self.nz+1:,:]).transpose()/(2*epsilon)
        else:
          noise.copy_(a.unsqueeze(2).unsqueeze(3))
          noisev = Variable(noise, requires_grad=True)
          fakeIms, _ = self.netG(noisev)
  #        I = fake.data.cpu().numpy() #memory problems?
          if deepFeatures is not None:
            fake = deepFeatures(fakeIms)
          else:
            fake = fakeIms
          # Insert deep features here
          fake = fake.view(1,-1)

          for k in range(nX):
            if k%1000 == 0:
              self.log("bprop iter {}".format(k))
            self.netG.zero_grad()
            if deepFeatures is not None:
              deepFeatures.zero_grad()
            fake[0,k].backward(retain_variables=True) #Not sure if retain variables is necessary
            J[k] = noisev.grad.data.cpu().numpy().squeeze()

        fakeIms = fakeIms.data.cpu().numpy()
        images[i] = fakeIms[0,:].reshape(self.nc, self.imSize, self.imSize)
        if deepFeatures is not None:
          feats[i] = fake.data[0,:].cpu().numpy()
        
        R = np.linalg.qr(J, mode='r')
        Z = a.cpu().numpy()
        dummy = R.diagonal().copy()
        jacob[i] = dummy.copy() # No modification yet
        dummy[np.where(np.abs(dummy) < 1e-20)] = 1
        probs[i] = -log_const*0.5*np.sum(Z**2)+gauss_const - np.log10(np.abs(dummy)).sum()

    
		# What about codes?
    if deepFeatures is not None:
      allData = {'images': images.astype(np.float32),
                 'feats': feats.astype(np.float32),
                 'prob': probs.astype(np.float32), #no s to be consistent
                 'jacob': jacob.astype(np.float32)}
    else:
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
      self.save(self.netG.state_dict(), self.netGkey, instance=checkpointNum, saver='torch')
      self.save(self.netD.state_dict(), self.netDkey, instance=checkpointNum, saver='torch')
      self.save((self.images,self.errG, self.errD), 'ganState', instance=checkpointNum, saver='pickle')
    else:
      self.save(self.netG.state_dict(), self.netGkey, saver='torch')
      self.save(self.netD.state_dict(), self.netDkey, saver='torch')
      self.save((self.images,self.errG, self.errD), 'ganState', saver='pickle')


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

