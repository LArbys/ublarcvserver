import os,sys,time
from array import array

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# sparse submanifold convnet library
import sparseconvnet as scn

# ROOT
import ROOT as rt
from larcv import larcv

from func_intersect_ub import IntersectUB

class SparseLArFlow3DConsistencyLoss(nn.Module):
    """
    Loss for LArFlow which combines L2 loss between predicted and true flow,
    but also an additional loss from 3D inconsistency in the Y2U and Y2V flows.

    The consistency loss is measured by taking the L2 distance between the
    3D point predicted by the Y2U and Y2V flows. Because we assume
    same tick correspondences only (fixing x) and flow from the Y plane (fixing z),
    we really are only measuring the distance in the y-detector dimension.
    """
    def __init__(self, nrows, ncols, calc_consistency=True,
                 intersectiondata=None, larcv_version=None,
                 nsource_wires=3456, ntarget_wires=2400, goodrange=None,
                 return_pos_images=False, reduce=True):
        super(SparseLArFlow3DConsistencyLoss,self).__init__()
        """
        inputs
        ------
        nrows [int]: image rows
        ncols [int]" image cols
        intersectiondata [str]: path to rootfile which stores a table
                                for what Y-position corresponds to some wire crossing.
                                if none, only the flow prediction loss is used
        larcv_version [None or int]: 1 or 2 for larcv 1 or 2
        nsource_wires [int]: don't remember
        ntarget_wires [int]: don't remember
        """
        self.calc_consistency = calc_consistency
        if self.calc_consistency:
            IntersectUB.load_intersection_data(intersectiondatafile,larcv_version=larcv_version,
                                               nsource_wires=nsource_wires,ntarget_wires=ntarget_wires)
            IntersectUB.set_img_dims( nrows, ncols )

        if goodrange is not None:
            self.goodrange_t = torch.zeros( (ncols,nrows), dtype=torch.float )
            self.goodrange_t[goodrange[0]:goodrange[1],:] = 1.0
        else:
            self.goodrange_t = None
        self._return_pos_images = return_pos_images
        self._reduce = reduce

        #self.truth1_input = scn.InputLayer(2,(nrows,ncols),mode=0)
        #self.truth2_input = scn.InputLayer(2,(nrows,ncols),mode=0) 
        self.outlayer1 = scn.OutputLayer(2)
        self.outlayer2 = scn.OutputLayer(2)


    def forward(self,coord, flow1_predict,flow2_predict,flow1_truth,flow2_truth,
                source_originx=0, targetu_originx=0, targetv_originx=0):

        """
        input
        -----
        coord_t       [SparseConvTensor (N,3)]: list of (row,col,batch)
        flow1_predict [SparseConvTensor (N,1)]: predicted flow for Y2U.
        flow2_predict [SparseConvTensor (N,1)]: predicted flow for Y2V. 
        flow1_truth   [tensor (N,1)]: predicted flow for Y2U.
        flow2_truth   [tensor (N,1)]: predicted flow for Y2V. Coordinates is where we is.
        """

        if self.calc_consistency:
            posyz_target1_t,posyz_target2_t = \
            IntersectUB.apply( flow1_predict, flow2_predict,
                            source_originx, targetu_originx, targetv_originx )

        flowout1 = self.outlayer1(flow1_predict)
        flowout2 = self.outlayer2(flow2_predict)

        mask1 = torch.ones( flow1_truth.shape, dtype=torch.float ).to(flow1_truth.device)
        mask2 = torch.ones( flow2_truth.shape, dtype=torch.float ).to(flow1_truth.device)
        #print "mask1: ",mask1.shape,"raw sum=",mask1.detach().sum()
        #print "mask2: ",mask2.shape,"raw sum=",mask2.detach().sum()
        mask1[ torch.ne(flow1_truth,0) ] = 0
        mask2[ torch.ne(flow2_truth,0) ] = 0
        #print "mask1: ",mask1.shape,"select sum=",mask1.detach().sum()
        #print "mask2: ",mask2.shape,"select sum=",mask2.detach().sum()

        #print posyz_target1_t.size()," vs. mask=",mask.size()
        if self.calc_consistency:
            posyz_target1_t *= mask
            posyz_target2_t *= mask

        # flow prediction loss
        flow1err = (flow1_truth-flowout1)*mask1
        flow2err = (flow2_truth-flowout2)*mask2
        if mask1.sum()>0:
            flow1err = flow1err*flow1err/mask1.sum()
        if mask2.sum()>0:
            flow2err = flow2err*flow2err/mask2.sum()

        #print "posyz 1: ",np.argwhere( np.isnan( posyz_target1_t.detach().cpu().numpy() ) )
        #print "posyz 2: ",np.argwhere( np.isnan( posyz_target2_t.detach().cpu().numpy() ) )

        # calculate the squared difference between the points
        #diff_yz = posyz_target1_t-posyz_target2_t # take diff
        #l2 = diff_yz[:,0,:,:]*diff_yz[:,0,:,:] + diff_yz[:,1,:,:]*diff_yz[:,1,:,:] # square

        #print "diffyz: ",np.argwhere( np.isnan( diff_yz.detach().cpu().numpy() ) )
        #print "mask.sum: ",np.argwhere( np.isnan( mask.sum().detach().cpu().numpy() ) )
        if self._reduce:
            l2flow = flow1err.sum() + flow2err.sum()
        else:
            l2flow = flow1err+flow2err
        #    loss = l2.sum()
        #    # loss is the mean loss per non-masked pixel
        #    if mask.sum()>0:
        #        loss = l2.sum()/mask.sum() # divide by number of non-masked pixels
        #else:
        #    loss = l2

        #if not self._return_pos_images:
        #    # default
        #    return loss
        #else:
        #    # for debug typically
        #    return loss,posyz_target1_t,posyz_target2_t
        return l2flow


if __name__ == "__main__":

    nrows = 1008
    ncols  = 3456
    loss_w_3dconsistency = SparseLArFlow3DConsistencyLoss( nrows, ncols )

    # test data
    inputfile = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    batchsize = 1
    nworkers  = 1
    tickbackward = True
    readonly_products=( ("wiremc",larcv.kProductImage2D),
                        ("larflow",larcv.kProductImage2D) )

    nentries = 1


    feeder = load_larflow_larcvdata( "larflowsparsetest", inputfile, batchsize, nworkers,
                                     tickbackward=tickbackward,readonly_products=readonly_products )
    tstart = time.time()

    for n in xrange(nentries):
        batch = feeder.get_batch_dict()
        print "ENTRY[",n,"] from ",batch["feeder"]

        coord_np = data["srcpix"]
