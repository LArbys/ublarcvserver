import os,sys
import torch
import torch.nn as nn
import sparseconvnet as scn

from layer_utils import create_resnet_layer

class SparseEncoder(nn.Module):
    """
    The encoder involves a series of layers which
    1) first apply a series of residual convolution blocks (using 3x3 kernels)
    2) then apply a strided convolution to downsample the image
       right now, a standard stride of 2 is used
    """

    def __init__( self, name, nreps, ninputchs, chs_per_layer):
        nn.Module.__init__(self)
        """
        inputs
        ------
        name      [str] to name the encoder
        nreps     [int] number of times residual block is repeated per layer
        ninputchs [int] input channels to entire encoder module
        chs_per_layer [list of int] output channels for each layer
        """
        # store variables and configuration
        self.chs_per_layer = chs_per_layer
        self.inputchs = ninputchs
        self.residual_blocks = True # always uses residual blocks
        self.dimension = 2 # always for 2D images
        self._layers = []  # stores layers inside this module
        self._name = name  # name of this instance
        self._verbose = False

        # create the individual layers
        for ilayer,noutputchs in enumerate(self.chs_per_layer):
            if ilayer>0:
                # use last num output channels
                # in previous layer for num input ch
                ninchs = self.chs_per_layer[ilayer-1]
            else:
                # for first layer, use the provided number of chanels
                ninchs = self.inputchs

            # create the encoder layer
            layer = self.make_encoder_layer(ninchs,noutputchs,nreps)
            # store it
            self._layers.append(layer)
            # create an attribute for the module so pytorch
            # can know the components in this module
            setattr(self,"%s_enclayer%d"%(name,ilayer),layer)

    def set_verbose(self,verbose=True):
        """ control output of debugging info. mainly shapes of output by layers"""
        self._verbose = verbose

    def make_encoder_layer(self,ninputchs,noutputchs,nreps,
                           leakiness=0.01,downsample=[2, 2]):
        """
        inputs
        ------
        ninputchs [int]: number of features going into layer
        noutputchs [int]: number of features output by layer
        nreps [int]: number of times residual modules repeated
        leakiness [int]: leakiness of LeakyReLU layers
        downsample [length 2 list of int]: stride in [height,width] dims

        outputs
        -------
        scn.Sequential module with resnet and downsamping layers
        """
        encode_blocks = create_resnet_layer(nreps,ninputchs,noutputchs,
                                            downsample=downsample)
        if downsample is not None:
            # if we specify downsize factor for each dimension, we apply
            # it to the output of the residual layers
            encode_blocks.add(scn.BatchNormLeakyReLU(noutputchs,leakiness=leakiness))
            encode_blocks.add(scn.Convolution(self.dimension, noutputchs, noutputchs,
                                              downsample[0], downsample[1], False))
        return encode_blocks

    def forward(self,x):
        """
        run the layers on the input

        inputs
        ------
        x [scn.SparseManifoldTensor]: input tensor

        outputs
        -------
        list of scn.SparseManifoldTensor: one tensor for each
           encoding layer. all returned for use in skip connections
           in the SparseDecoding modules
        """
        layerout = []
        for ilayer,layer in enumerate(self._layers):
            if ilayer==0:
                inputx = x
            else:
                inputx = layerout[-1]
            out = layer(inputx)
            if self._verbose:
                print "[%s] Encode Layer[%d]: "%(self._name,ilayer),
                print inputx.features.shape,inputx.spatial_size,
                print "-->",out.features.shape,out.spatial_size
            layerout.append( out )
        return layerout

