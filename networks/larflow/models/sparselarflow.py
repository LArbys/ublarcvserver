import sys
import torch
import torch.nn as nn
import sparseconvnet as scn
import time
from sparse_encoder import SparseEncoder
from sparse_decoder import SparseDecoder

from layer_utils import create_resnet_layer

class SparseLArFlow(nn.Module):
    """
    Sparse Submanifold implementation of LArFlow
    """
    def __init__(self, inputshape, reps, nin_features, nout_features, nplanes):
        nn.Module.__init__(self)
        """
        inputs
        ------
        inputshape [list of int]: dimensions of the matrix or image
        reps [int]: number of residual modules per layer (for both encoder and decoder)
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that feed into the regression layer
        nplanes [int]: the depth of the U-Net
        """
        # set parameters
        self.dimensions = 2 # not playing with 3D for now

        # input shape: LongTensor, tuple, or list. Handled by InputLayer
        # size of each spatial dimesion
        self.inputshape = inputshape
        if len(self.inputshape)!=self.dimensions:
            raise ValueError("expected inputshape to contain size of 2 dimensions only."
                             +"given %d values"%(len(self.inputshape)))

        # mode variable: how to deal with repeated data
        self.mode = 0

        # nfeatures
        self.nfeatures = nin_features
        self.nout_features = nout_features

        # plane structure
        self.nPlanes = [ self.nfeatures*2**(n+1) for n in xrange(nplanes) ]
        print self.nPlanes

        # repetitions (per plane)
        self.reps = reps

        # residual blocks
        self.residual_blocks = True

        # need encoder for both source and target
        # then cat tensor
        # and produce one decoder for flow, another decoder for visibility

        # model:
        # input
        self.src_inputlayer  = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        self.tar1_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        self.tar2_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)

        # stem
        self.src_stem  = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
        self.tar1_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
        self.tar2_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)

        # encoders
        self.source_encoder  = SparseEncoder( "src",  self.reps, self.nfeatures, self.nPlanes )
        self.target1_encoder = SparseEncoder( "tar1", self.reps, self.nfeatures, self.nPlanes )
        self.target2_encoder = SparseEncoder( "tar2", self.reps, self.nfeatures, self.nPlanes )

        # concat
        self.join_enclayers = []
        for ilayer in xrange(len(self.nPlanes)):
            self.join_enclayers.append(scn.JoinTable())
            setattr(self,"join_enclayers%d"%(ilayer),self.join_enclayers[ilayer])

        # calculate decoder planes
        self.decode_layers_inchs  = []
        self.decode_layers_outchs = []
        for ilayer,enc_outchs in enumerate(reversed(self.nPlanes)):
            self.decode_layers_inchs.append( 4*enc_outchs if ilayer>0 else 3*enc_outchs )
            self.decode_layers_outchs.append( self.nPlanes[-(1+ilayer)]/2 )
        print "decode in chs: ",self.decode_layers_inchs
        print "decode out chs: ",self.decode_layers_outchs

        # decoders
        self.flow1_decoder = SparseDecoder( "flow1", self.reps,
                                            self.decode_layers_inchs,
                                            self.decode_layers_outchs )
        self.flow2_decoder = SparseDecoder( "flow2", self.reps,
                                            self.decode_layers_inchs,
                                            self.decode_layers_outchs )

        # last deconv concat
        self.flow1_concat = scn.JoinTable()
        self.flow2_concat = scn.JoinTable()

        # final feature set convolution
        flow_resblock_inchs = 3*self.nfeatures + self.decode_layers_outchs[-1]
        self.flow1_resblock = create_resnet_layer(self.reps,
                                                  flow_resblock_inchs,self.nout_features)
        self.flow2_resblock = create_resnet_layer(self.reps,
                                                  flow_resblock_inchs,self.nout_features)

        # regression layer
        self.flow1_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)
        self.flow2_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)


    def forward(self, coord_t, src_feat_t, tar1_feat_t, tar2_feat_t, batchsize):
        """
        run the network

        inputs
        ------
        coord_flow1_t [ (N,3) Torch Tensor ]: list of (row,col,batchid) N pix coordinates
        src_feat_t  [ (N,) torch tensor ]: list of pixel values for source image
        tar1_feat_t [ (N,) torch tensor ]: list of pixel values for target 1 image
        tar2_feat_t [ (N,) torch tensor ]: list of pixel values for target 2 image
        batchsize [int]: batch size

        outputs
        -------
        [ (N,) torch tensor ] flow values to target 1
        [ (N,) torch tensor ] flow values to target 2
        """
        srcx = ( coord_t, src_feat_t,  batchsize )
        tar1 = ( coord_t, tar1_feat_t, batchsize )
        tar2 = ( coord_t, tar2_feat_t, batchsize )

        # source encoder
        srcx = self.src_inputlayer(srcx)
        srcx = self.src_stem(srcx)
        srcout_v = self.source_encoder(srcx)

        tar1 = self.tar1_inputlayer(tar1)
        tar1 = self.tar1_stem(tar1)
        tar1out_v = self.target1_encoder(tar1)

        tar2 = self.tar1_inputlayer(tar2)
        tar2 = self.tar1_stem(tar2)
        tar2out_v = self.target2_encoder(tar2)

        # concat features from all three planes
        joinout = []
        for _src,_tar1,_tar2,_joiner in zip(srcout_v,tar1out_v,tar2out_v,self.join_enclayers):
            joinout.append( _joiner( (_src,_tar1,_tar2) ) )

        # Flow 1: src->tar1
        # ------------------
        # use 3-plane features to make flow features
        flow1 = self.flow1_decoder( joinout )

        # concat stem out with decoder out
        flow1 = self.flow1_concat( (flow1,srcx,tar1,tar2) )

        # last feature conv layer
        flow1 = self.flow1_resblock( flow1 )

        # finally, 1x1 conv layer from features to flow value
        flow1 = self.flow1_out( flow1 )

        # Flow 2: src->tar1
        # ------------------
        # use 3-plane features to make flow features
        flow2 = self.flow2_decoder( joinout )

        # concat stem out with decoder out
        flow2 = self.flow2_concat( (flow2,srcx,tar1,tar2) )

        # last feature conv layer
        flow2 = self.flow2_resblock( flow2 )

        # finally, 1x1 conv layer from features to flow value
        flow2 = self.flow2_out( flow2 )

        return flow1,flow2

if __name__ == "__main__":
    """
    here we test/debug the network and loss function
    we can use a random matrix that mimics our sparse lartpc images
      or actual images from the loader.
    """

    nrows     = 1024
    ncols     = 3456
    sparsity  = 0.01
    device    = torch.device("cpu")
    #device    = torch.device("cuda")
    ntrials   = 1
    batchsize = 1
    use_random_data = False
    test_loss = True
    
    model = SparseLArFlow( (nrows,ncols), 2, 16, 16, 4 ).to(device)
    model.eval()
    #print model

    npts = int(nrows*ncols*sparsity)
    print "for (%d,%d) and average sparsity of %.3f, expected npts=%d"%(nrows,ncols,sparsity,npts)

    if not use_random_data:
        from larcv import larcv
        from sparselarflowdata import load_larflow_larcvdata
        #inputfile    = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
        inputfile = "out_sparsified.root"
        nworkers     = 3
        tickbackward = True
        #ro_products  = ( ("wiremc",larcv.kProductImage2D),
        #                 ("larflow",larcv.kProductImage2D) )
        ro_products = None
        dataloader   = load_larflow_larcvdata( "larflowsparsetest", inputfile,
                                                batchsize, nworkers,
                                                tickbackward=tickbackward,
                                                readonly_products=ro_products )

    if test_loss:
        from loss_sparse_larflow import SparseLArFlow3DConsistencyLoss
        consistency_loss = SparseLArFlow3DConsistencyLoss(1024, 3456,
                                                          larcv_version=1,
                                                          calc_consistency=False)

    # random points from a hypothetical (nrows x ncols) image
    dtforward = 0
    dtdata = 0
    for itrial in xrange(ntrials):

        # random data
        tdata = time.time()
        if use_random_data:            
            """ randomly filled matrix """
            import numpy as np
            
            coords = np.zeros( (npts,2), dtype=np.int )
            coords[:,0] = np.random.randint( 0, nrows, npts )
            coords[:,1] = np.random.randint( 0, ncols, npts )
            srcx = np.random.random( (npts,1) ).astype(np.float32)
            tar1 = np.random.random( (npts,1) ).astype(np.float32)
            tar2 = np.random.random( (npts,1) ).astype(np.float32)
            truth1 = np.random.random( (npts,1) ).astype(np.float32)
            truth2 = np.random.random( (npts,1) ).astype(np.float32)

            coord_t   = torch.from_numpy(coords).to(device)
            srcpix_t  = torch.from_numpy(srcx).to(device)
            tarpix_flow1_t = torch.from_numpy(tar1).to(device)
            tarpix_flow2_t = torch.from_numpy(tar2).to(device)
            truth_flow1_t = torch.from_numpy(truth1).to(device)
            truth_flow2_t = torch.from_numpy(truth2).to(device) 
        else:
            """ data from actual larflow larcv file """
            datadict = dataloader.get_tensor_batch(device)
            coord_t  = datadict["coord"]
            srcpix_t = datadict["src"]
            tarpix_flow1_t = datadict["tar1"]
            tarpix_flow2_t = datadict["tar2"]
            truth_flow1_t  = datadict["flow1"]
            truth_flow2_t  = datadict["flow2"]

        dtdata += time.time()-tdata

        tforward = time.time()
        print "coord-shape: flow1=",coord_t.shape
        print "src feats-shape: ",srcpix_t.shape
        if truth_flow1_t is not None:
            print "truth flow1: ",truth_flow1_t.shape
        predict1_t,predict2_t = model( coord_t, srcpix_t,
                            tarpix_flow1_t, tarpix_flow2_t,
                            batchsize )
        dtforward += time.time()-tforward

        if test_loss:
            tloss = time.time()
            loss = consistency_loss(coord_t, predict1_t, predict2_t,
                                    truth_flow1_t, truth_flow2_t)
            print "loss: ",loss.detach().cpu().item()

        #print "modelout: flow1=[",out1.features.shape,out1.spatial_size,"]"

    print "ave. data time o/ %d trials: %.2f secs"%(ntrials,dtdata/ntrials)
    print "ave. forward time o/ %d trials: %.2f secs"%(ntrials,dtforward/ntrials)
