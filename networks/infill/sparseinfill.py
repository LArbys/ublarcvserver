# Imports
import os,sys,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/SparseConvNet")
import sparseconvnet as scn

import time
import math
import numpy as np

class SparseInfill(nn.Module):
    def __init__(self, inputshape, reps, nin_features, nout_features, nplanes,show_sizes):
        nn.Module.__init__(self)
        """
        inputs
        ------
        inputshape [list of int]: dimensions of the matrix or image
        reps [int]: number of residual modules per layer (for both encoder and decoder)
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that feed into the regression layer
        nPlanes [int]: the depth of the U-Net
        show_sizes [bool]: if True, print sizes while running forward
        """
        self._mode = 0
        self._dimension = 2
        self._inputshape = inputshape
        if len(self._inputshape)!=self._dimension:
            raise ValueError("expected inputshape to contain size of 2 dimensions only."
                             +"given %d values"%(len(self._inputshape)))
        self._reps = reps
        self._nin_features = nin_features
        self._nout_features = nout_features
        self._nplanes = [nin_features, 2*nin_features, 3*nin_features, 4*nin_features, 5*nin_features]
        self._show_sizes = show_sizes

        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(self._dimension, self._inputshape, mode=self._mode)).add(
           scn.SubmanifoldConvolution(self._dimension, 1, self._nin_features, 3, False)).add(
           scn.UNet(self._dimension, self._reps, self._nplanes, residual_blocks=True, downsample=[2,2])).add(
           scn.BatchNormReLU(self._nin_features)).add(
           scn.OutputLayer(self._dimension))

        self.input = scn.InputLayer(self._dimension, self._inputshape, mode=self._mode)
        self.conv1 = scn.SubmanifoldConvolution(self._dimension, 1, self._nin_features, 3, False)
        self.unet = scn.UNet(self._dimension, self._reps, self._nplanes, residual_blocks=True, downsample=[2,2])
        self.batchnorm = scn.BatchNormReLU(self._nin_features)
        self.output = scn.OutputLayer(self._dimension)
        self.conv2 = scn.SubmanifoldConvolution(self._dimension, self._nin_features, 1, 3, False)

    def forward(self,coord_t,input_t,batchsize):
        if self._show_sizes:
            print "coord_t ",coord_t.shape
            print "input_t ",input_t.shape
        x=(coord_t,input_t,batchsize)
        x=self.input(x)
        if self._show_sizes:
            print "inputlayer: ",x.features.shape
        x=self.conv1(x)
        if self._show_sizes:
            print "conv1: ",x.features.shape
        x=self.unet(x)
        if self._show_sizes:
            print "unet: ",x.features.shape
        x=self.batchnorm(x)
        if self._show_sizes:
            print "batchnorm: ",x.features.shape
        x=self.conv2(x)
        if self._show_sizes:
            print "conv2: ",x.features.shape
        x=self.output(x)
        if self._show_sizes:
            print "output: ",x.shape
        return x
