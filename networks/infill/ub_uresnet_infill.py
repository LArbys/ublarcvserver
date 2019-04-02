import os,sys
import torch as torch
import torch.nn as nn
import math

from common_layers import *

###########################################################
#
# U-ResNet
# U-net witih ResNet modules
#
# Semantic segmentation network used by MicroBooNE
# to label track/shower pixels
#
# resnet implementation from pytorch.torchvision module
# U-net from (cite)
#
# meant to be copy of caffe version
# except:
#   -- no group deconvolutions
#   -- last layer removes relu and batchnorm
#
#
###########################################################



class UResNetInfill(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, inplanes=16, final_conv_kernels=16, showsizes=False):
        self.inplanes =inplanes
        super(UResNetInfill, self).__init__()

        self._showsizes = showsizes # print size at each layer

        # Encoder

        # stem
        # one big stem
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=1, padding=3, bias=True) # initial conv layer
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d( 3, stride=2, padding=1 )

        self.enc_layer1 = self._make_encoding_layer( self.inplanes*1,  self.inplanes*2,  stride=1) # 16->32
        self.enc_layer2 = self._make_encoding_layer( self.inplanes*2,  self.inplanes*4,  stride=2) # 32->64
        self.enc_layer3 = self._make_encoding_layer( self.inplanes*4,  self.inplanes*8,  stride=2) # 64->128
        self.enc_layer4 = self._make_encoding_layer( self.inplanes*8,  self.inplanes*16, stride=2) # 128->256
        self.enc_layer5 = self._make_encoding_layer( self.inplanes*16, self.inplanes*32, stride=2) # 256->512

        self.dec_layer5 = self._make_decoding_layer( self.inplanes*32,  self.inplanes*16, self.inplanes*16 ) # 512->256
        self.dec_layer4 = self._make_decoding_layer( self.inplanes*16,  self.inplanes*8,  self.inplanes*8  ) # 256->128
        self.dec_layer3 = self._make_decoding_layer( self.inplanes*8,   self.inplanes*4,  self.inplanes*4  ) # 128->64
        self.dec_layer2 = self._make_decoding_layer( self.inplanes*4,   self.inplanes*2,  self.inplanes*2  ) # 64->32
        self.dec_layer1 = self._make_decoding_layer( self.inplanes*2,   self.inplanes,    self.inplanes    ) # 32->16

        # final conv stem (7x7) = (3x3)^3
        self.nkernels = final_conv_kernels
        self.conv10 = nn.Conv2d(self.inplanes, self.nkernels, kernel_size=7, stride=1, padding=3, bias=True) # initial conv layer
        self.bn10   = nn.BatchNorm2d(self.nkernels)
        self.relu10 = nn.ReLU(inplace=False)

        self.conv11 = nn.Conv2d(self.nkernels, num_classes, kernel_size=7, stride=1, padding=3, bias=True) # initial conv layer
        #self.bn11   = nn.BatchNorm2d(num_classes)
        #self.relu11 = nn.ReLU(inplace=True)


        # we use log softmax in order to more easily pair it with
        # self.softmax = nn.LogSoftmax(dim=1) # should return [b,c=3,h,w], normalized over, c dimension

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_encoding_layer(self, inplanes, planes, stride=2):

        return DoubleResNet(inplanes,planes,stride=stride)

    def _make_decoding_layer(self, inplanes, deconvplanes, resnetplanes ):
        return ConvTransposeLayer( inplanes, deconvplanes, resnetplanes )

    def forward(self, x):

        if self._showsizes:
            print "input: ",x.size()," is_cuda=",x.is_cuda

        # stem
        x  = self.conv1(x)
        x  = self.bn1(x)
        x0 = self.relu1(x)
        x = self.pool1(x0)


        if self._showsizes:
            print "after conv1, x0: ",x0.size()

        x1 = self.enc_layer1(x)
        x2 = self.enc_layer2(x1)
        x3 = self.enc_layer3(x2)
        x4 = self.enc_layer4(x3)
        x5 = self.enc_layer5(x4)
        if self._showsizes:
            print "after encoding: "
            print "  x1: ",x1.size()
            print "  x2: ",x2.size()
            print "  x3: ",x3.size()
            print "  x4: ",x4.size()
            print "  x5: ",x5.size()

        x = self.dec_layer5(x5,x4)
        if self._showsizes:
            print "after decoding:"
            print "  dec5: ",x.size()," iscuda=",x.is_cuda

        x = self.dec_layer4(x,x3)
        if self._showsizes:
            print "  dec4: ",x.size()," iscuda=",x.is_cuda

        x = self.dec_layer3(x,x2)
        if self._showsizes:
            print "  dec3: ",x.size()," iscuda=",x.is_cuda

        x = self.dec_layer2(x,x1)
        if self._showsizes:
            print "  dec2: ",x.size()," iscuda=",x.is_cuda

        x = self.dec_layer1(x,x0)
        if self._showsizes:
            print "  dec1: ",x.size()," iscuda=",x.is_cuda

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)

        x = self.conv11(x)

        # x = self.softmax(x)
        # if self._showsizes:
        #     print "  softmax: ",x.size()

        return x
