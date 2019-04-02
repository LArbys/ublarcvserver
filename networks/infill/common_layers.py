import os,sys
import torch as torch
import torch.nn as nn
import math

# =========================================
# Wrappers around pytorch NN layers
#
# Building blocks for models
#
# ===========================================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
                     

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)        
        self.stride = stride

        self.bypass = None
        self.bnpass = None
        if inplanes!=planes or stride>1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bnpass = nn.BatchNorm2d(planes)
            
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)        

        if self.bypass is not None:
            outbp = self.bypass(x)
            outbp = self.bnpass(outbp)
            out += outbp
        else:
            out += x

        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1 ):
        super(Bottleneck, self).__init__()

        # residual path
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                               
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.stride = stride

        # if stride >1, then we need to subsamble the input
        if stride>1:
            self.shortcut = nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        else:
            self.shortcut = None
            

    def forward(self, x):

        if self.shortcut is None:
            bypass = x
        else:
            bypass = self.shortcut(x)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)

        residual = self.conv3(residual)
        residual = self.bn3(residual)

        out = bypass+residual
        out = self.relu(out)

        return out


class DoubleResNet(nn.Module):
    def __init__(self,inplanes,planes,stride=1):
        super(DoubleResNet,self).__init__()
        #self.res1 = Bottleneck(inplanes,planes,stride)
        #self.res2 = Bottleneck(  planes,planes,     1)
        self.res1 = BasicBlock(inplanes,planes,stride)
        self.res2 = BasicBlock(  planes,planes,     1)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        return out

class ConvTransposeLayer(nn.Module):
    def __init__(self,deconv_inplanes,deconv_outplanes,res_outplanes):
        super(ConvTransposeLayer,self).__init__()
        self.deconv = nn.ConvTranspose2d( deconv_inplanes, deconv_outplanes, kernel_size=4, stride=2, padding=1, bias=False )
        self.res    = DoubleResNet(res_outplanes+deconv_outplanes,res_outplanes,stride=1)
    def forward(self,x,skip_x):
        out = self.deconv(x,output_size=skip_x.size())
        # concat skip connections
        out = torch.cat( [out,skip_x], 1 )
        out = self.res(out)
        return out
