import torch
import torch.nn as nn
import sparseconvnet as scn

from layer_utils import create_resnet_layer

class SparseDecoder(nn.Module):
    """
    SparseDecoder class features layers containing:
    1) sequence of residual convolution modules
    2) upsampling using convtranspose
    3) skip connection from concating tensors

    Note that configuration of sparse decoder must be coordinated
    with setup of SparseEncoder in order to properly handle
    skip-connections.
    """
    def __init__( self, name, nreps, inputchs_per_layer, outputchs_per_layer ):
        nn.Module.__init__(self)
        """
        inputs
        ------
        name [str]
        nreps [int]
        inputchs_per_layer  [list of ints]
        outputchs_per_layer [list of ints]
        """
        self.inputchs_per_layer  = inputchs_per_layer
        self.outputchs_per_layer = outputchs_per_layer
        self.residual_blocks = True
        self.dimension = 2
        self._deconv_layers = []
        self._joiner = []
        self._name = name
        self._verbose = False

        layer_chs = zip(self.inputchs_per_layer,self.outputchs_per_layer)
        for ilayer,(ninputchs,noutputchs) in enumerate(layer_chs):
            if ilayer+1<len(outputchs_per_layer):
                islast = False
            else:
                islast = True
            deconv,joiner = self.make_decoder_layer(ilayer,ninputchs,
                                                    noutputchs,nreps,
                                                    islast=islast)
            self._deconv_layers.append(deconv)
            self._joiner.append(joiner)

    def make_decoder_layer(self, ilayer, ninputchs, noutputchs, nreps,
                           leakiness=0.01,downsample=[2, 2],islast=False):
        """
        defines two layers:
          1) the deconv layer pre-concat
          2) residual blocks post-concat

        inputs
        ------
        ilayer     [int]: layer ID 
        ninputchs  [int]: number of features going into layer
        noutputchs [int]: number of features output by layer
        nreps      [int]: number of times residual modules should repeat
        leakiness  [float]: leakiness of LeakyReLU activiation functions
        downsample [list of ints size 2]: upsampling factor in weight and height
        islast     [bool]: last decoder layer does not have skip connection
        """

        # resnet block
        decode_blocks = create_resnet_layer(nreps,ninputchs,2*noutputchs,
                                            downsample=downsample)

        # deconv
        decode_blocks.add(scn.BatchNormLeakyReLU(2*noutputchs,leakiness=leakiness))
        decode_blocks.add(scn.Deconvolution(self.dimension, 2*noutputchs, noutputchs,
                                            downsample[0], downsample[1], False))
        setattr(self,"deconv%d"%(ilayer),decode_blocks)
        if self._verbose:
            print "DecoderLayer[",ilayer,"] inputchs[",ninputchs,
            print " -> resout[",2*noutputchs,"] -> deconv output[",noutputchs,"]"


        if not islast:
            # joiner for skip connections
            joiner = scn.JoinTable()
            setattr(self,"skipjoin%d"%(ilayer),joiner)
        else:
            joiner = None
        return decode_blocks,joiner

    def forward(self,encoder_layers):
        """
        inputs
        ------
        encoder_layers [list of sparsetensors]: output of SparseEncoder layer

        outputs
        -------
        [ (N,1) SparseConvMatrix ] features contain flow predictions
        """
        layerout = None
        for ilayer,(deconvl,joiner) in enumerate(zip(self._deconv_layers,self._joiner)):
            if ilayer==0:
                # first layer, input tensor comes from last encoder layer
                inputx = encoder_layers[-(1+ilayer)]
            else:
                # otherwise, get the last tensor that was created
                inputx = layerout
            #print "decoder-input[",ilayer,"] input=",inputx.features.shape,inputx.spatial_size

            # residual + upsample
            out = deconvl(inputx)
            #print "decode-deconv[",ilayer,"]: outdecode=",out.features.shape,out.spatial_size

            # concat if not last layer, for last layer, joiner will be None
            if joiner is not None:
                # get next encoder layer
                catlayer  = encoder_layers[-(2+ilayer)]
                #print "decode-concat[",ilayer,"]: appending=",
                #print catlayer.features.shape,catlayer.spatial_size

                # concat
                out   = joiner( (out,catlayer) )

            if self._verbose:
                print "[%s] Decode Layer[%d]: "%(self._name,ilayer),
                print inputx.features.shape,inputx.spatial_size,
                print "-->",out.features.shape,out.spatial_size
            layerout = out
            
        return layerout

