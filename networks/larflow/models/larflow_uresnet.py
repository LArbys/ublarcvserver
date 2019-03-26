import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo

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
# 
###########################################################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
                     

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride = stride

        self.bypass = None
        if inplanes!=planes or stride>1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.bypass is not None:
            outbp = self.bypass(x)
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
        self.relu = nn.ReLU(inplace=True)
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

class PreactivationBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1 ):
        super(Preactivation, self).__init__()

        # residual path
        self.bn1   = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)

        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

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
    
    

class DoubleResNet(nn.Module):
    def __init__(self,Block,inplanes,planes,stride=1):
        super(DoubleResNet,self).__init__()
        self.res1 = Block(inplanes,planes,stride)
        self.res2 = Block(  planes,planes,     1)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        return out

class ConvTransposeLayer(nn.Module):
    def __init__(self,deconv_inplanes,skip_inplanes,deconv_outplanes,res_outplanes):
        super(ConvTransposeLayer,self).__init__()
        self.deconv = nn.ConvTranspose2d( deconv_inplanes, deconv_outplanes, kernel_size=4, stride=2, padding=1, bias=False )
        self.res    = DoubleResNet(BasicBlock,deconv_outplanes+skip_inplanes,res_outplanes,stride=1)
    def forward(self,x,skip_x):
        out = self.deconv(x,output_size=skip_x.size())
        # concat skip connections
        out = torch.cat( [out,skip_x], 1 )
        out = self.res(out)
        return out
    
class LArFlowUResNet(nn.Module):

    def __init__(self, num_classes=3, input_channels=3, inplanes=16, showsizes=False, use_visi=True):
        self.inplanes =inplanes
        super(LArFlowUResNet, self).__init__()

        self._showsizes = showsizes # print size at each layer
        self.use_visi = use_visi
        
        # Encoder

        # stem
        # one big stem
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=1, padding=3, bias=True) # initial conv layer
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d( 3, stride=2, padding=1 )
        
        self.enc_layer1 = self._make_encoding_layer( self.inplanes*1,  self.inplanes*2,  stride=1) # 16->32
        self.enc_layer2 = self._make_encoding_layer( self.inplanes*2,  self.inplanes*4,  stride=2) # 32->64
        self.enc_layer3 = self._make_encoding_layer( self.inplanes*4,  self.inplanes*8,  stride=2) # 64->128
        self.enc_layer4 = self._make_encoding_layer( self.inplanes*8,  self.inplanes*16, stride=2) # 128->256
        self.enc_layer5 = self._make_encoding_layer( self.inplanes*16, self.inplanes*32, stride=2) # 256->512

        # decoding flow
        #self.num_final_flow_features = self.inplanes
        self.num_final_flow_features = self.inplanes
        self.flow_dec_layer5 = self._make_decoding_layer( self.inplanes*32*2,  self.inplanes*16, self.inplanes*16, self.inplanes*16 ) # 512->256
        self.flow_dec_layer4 = self._make_decoding_layer( self.inplanes*16,    self.inplanes*8,  self.inplanes*8,  self.inplanes*8  ) # 256->128
        self.flow_dec_layer3 = self._make_decoding_layer( self.inplanes*8,     self.inplanes*4,  self.inplanes*4,  self.inplanes*4  ) # 128->64
        self.flow_dec_layer2 = self._make_decoding_layer( self.inplanes*4,     self.inplanes*2,  self.inplanes*2,  self.inplanes*2  ) # 64->32
        #self.flow_dec_layer1 = self._make_decoding_layer( self.inplanes*2,     self.inplanes,    self.inplanes    ) # 32->16
        self.flow_dec_layer1 = self._make_decoding_layer( self.inplanes*2,     self.inplanes,    self.inplanes, self.num_final_flow_features ) # 32->200

        # decoding matchability
        if self.use_visi:
            self.visi_dec_layer5 = self._make_decoding_layer( self.inplanes*32*2,  self.inplanes*16, self.inplanes*16, self.inplanes*16 ) # 512->256
            self.visi_dec_layer4 = self._make_decoding_layer( self.inplanes*16,    self.inplanes*8,  self.inplanes*8,  self.inplanes*8  ) # 256->128
            self.visi_dec_layer3 = self._make_decoding_layer( self.inplanes*8,     self.inplanes*4,  self.inplanes*4,  self.inplanes*4  ) # 128->64
            self.visi_dec_layer2 = self._make_decoding_layer( self.inplanes*4,     self.inplanes*2,  self.inplanes*2,  self.inplanes*2  ) # 64->32
            self.visi_dec_layer1 = self._make_decoding_layer( self.inplanes*2,     self.inplanes,    self.inplanes,    self.inplanes    ) # 32->16

        # 1x1 conv for flow
        self.flow_conv = nn.Conv2d( self.num_final_flow_features, 1, kernel_size=1, stride=1, padding=0, bias=True )

        # 1x1 conv for mathability
        if self.use_visi:
            self.visi_conv = nn.Conv2d( self.inplanes, 2, kernel_size=1, stride=1, padding=0, bias=True ) # 2 classes, 0=not vis, 1=vis
            self.visi_softmax = nn.LogSoftmax(dim=1)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoding_layer(self, inplanes, planes, stride=2):

        return DoubleResNet(BasicBlock,inplanes,planes,stride=stride)

    def _make_decoding_layer(self, inplanes, skipplanes, deconvplanes, resnetplanes ):
        return ConvTransposeLayer( inplanes, skipplanes, deconvplanes, resnetplanes )

    def encode(self,x):
        # stem
        x  = self.conv1(x)
        x  = self.bn1(x)
        x0 = self.relu1(x)
        x  = self.pool1(x0)

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
        return x5,x0,x1,x2,x3,x4
        
    def flow(self,merged_encode,x0,x1,x2,x3,x4):
        """ decoding to flow prediction """
        
        x = self.flow_dec_layer5(merged_encode,x4)
        if self._showsizes:
            print "after decoding:"
            print "  dec5: ",x.size()," iscuda=",x.is_cuda
            
        x = self.flow_dec_layer4(x,x3)
        if self._showsizes:
            print "  dec4: ",x.size()," iscuda=",x.is_cuda

        x = self.flow_dec_layer3(x,x2)
        if self._showsizes:
            print "  dec3: ",x.size()," iscuda=",x.is_cuda

        x = self.flow_dec_layer2(x,x1)
        if self._showsizes:
            print "  dec2: ",x.size()," iscuda=",x.is_cuda

        x = self.flow_dec_layer1(x,x0)
        if self._showsizes:
            print "  dec1: ",x.size()," iscuda=",x.is_cuda

        return x

    def visibility(self,merged_encode,x0,x1,x2,x3,x4):
        """ decoding to flow prediction """
        
        x = self.visi_dec_layer5(merged_encode,x4)
        if self._showsizes:
            print "after decoding:"
            print "  dec5: ",x.size()," iscuda=",x.is_cuda
            
        x = self.visi_dec_layer4(x,x3)
        if self._showsizes:
            print "  dec4: ",x.size()," iscuda=",x.is_cuda

        x = self.visi_dec_layer3(x,x2)
        if self._showsizes:
            print "  dec3: ",x.size()," iscuda=",x.is_cuda

        x = self.visi_dec_layer2(x,x1)
        if self._showsizes:
            print "  dec2: ",x.size()," iscuda=",x.is_cuda

        x = self.visi_dec_layer1(x,x0)
        if self._showsizes:
            print "  dec1: ",x.size()," iscuda=",x.is_cuda

        return x
            
    
    def forward(self, src, target):

        if self._showsizes:
            print "input: ",x.size()," is_cuda=",x.is_cuda

        src_encode,    s0, s1, s2, s3, s4  = self.encode(src)
        target_encode, t0, t1, t2, t3, t4  = self.encode(target)

        merged_encode = torch.cat( [target_encode,src_encode], 1 )

        flowout = self.flow( merged_encode, s0, s1, s2, s3, s4 )
        if self.use_visi:
            visiout = self.visibility( merged_encode, t0, t1, t2, t3, t4 )

        flow_predict = self.flow_conv( flowout )
        
        if self.use_visi:
            visi_predict = self.visi_conv( visiout )
            visi_predict = self.visi_softmax(visi_predict)
        else:
            visi_predict = None
        
        if self._showsizes:
            print "  softmax: ",x.size()
        
        return flow_predict,visi_predict


