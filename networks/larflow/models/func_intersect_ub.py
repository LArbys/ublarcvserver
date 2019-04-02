import os,time
import torch
from larcv import larcv
import numpy as np
import ROOT as rt
from array import array

class IntersectUB( torch.autograd.Function ):

    larcv_version = None
    dataloaded = False
    imgdimset = False

    @classmethod
    def load_intersection_data(cls,intersectiondatafile=None,larcv_version=None,nsource_wires=3456,ntarget_wires=2400):
        if intersectiondatafile is None:
            # set default
            if os.environ["LARCV_VERSION"].strip()=="1":
                intersectiondatafile = "../gen3dconsistdata/consistency3d_data_larcv1.root"
                cls.larcv_version = 1
            elif os.environ["LARCV_VERSION"].strip()=="2":
                intersectiondatafile = "../gen3dconsistdata/consistency3d_data_larcv2.root"
                cls.larcv_version = 2
            else:
                raise RuntimeError("Invalid LARCV_VERSION: {}".format(LARCV_VERSION))
        else:
            if larcv_version is None:
                raise ValueError("When specifiying data, need to specify larcv version")
            cls.larcv_version = larcv_version

        if not os.path.exists(intersectiondatafile):
            raise RuntimeError("could not find intersection data file: {}".format(intersectiondatafile))

        cls.nsource_wires = nsource_wires
        cls.ntarget_wires = ntarget_wires

        # intersection location (y,z) for (source,target) intersections
        cls.intersections_t  = torch.zeros( (2, 2, cls.nsource_wires, cls.ntarget_wires ) ).float()

        # fill intersection matrix (should make image2d instead of this loop fill

        if os.environ["LARCV_VERSION"]=="1":
            io = larcv.IOManager(larcv.IOManager.kREAD,"inersect3d",
                                    larcv.IOManager.kTickBackward)
            io.add_in_file(intersectiondata)
            io.initialize()
            ev_y2u = io.get_data(larcv.kProductImage2D,"y2u_intersect")
            if ev_y2u.Image2DArray().size()!=2:
                raise RuntimeError("Y2U intersection image2d vector should be len 2 (for detector y,z)")
            cls.intersections_t[0,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.Image2DArray()[0] ).reshape(cls.ntarget_wires,cls.nsource_wires).transpose((1,0)) )
            cls.intersections_t[0,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.Image2DArray()[1] ).reshape(cls.ntarget_wires,cls.nsource_wires).transpose((1,0)) )
            ev_y2v = io.get_data(larcv.kProductImage2D,"y2v_intersect")
            if ev_y2v.Image2DArray().size()!=2:
                raise RuntimeError("Y2V intersection image2d vector should be len 2 (for detector y,z)")
            cls.intersections_t[1,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.Image2DArray()[0] ).reshape(cls.ntarget_wires,cls.nsource_wires).transpose((1,0)) )
            cls.intersections_t[1,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.Image2DArray()[1] ).reshape(cls.ntarget_wires,cls.nsource_wires).transpose((1,0)) )
        elif os.environ["LARCV_VERSION"]=="2":
            io = larcv.IOManager()
            io.add_in_file(intersectiondatafile)
            io.initialize()
            ev_y2u = io.get_data("image2d","y2u_intersect")
            ev_y2v = io.get_data("image2d","y2v_intersect")
            cls.intersections_t[0,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.as_vector()[0] ).transpose((1,0)) )
            cls.intersections_t[0,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.as_vector()[1] ).transpose((1,0)) )
            cls.intersections_t[1,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.as_vector()[0] ).transpose((1,0)) )
            cls.intersections_t[1,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.as_vector()[1] ).transpose((1,0)) )

        cls.dataloaded = True

    @classmethod
    def set_img_dims(cls,nrows,ncols):

        cls.nrows = nrows
        cls.ncols = ncols

        # index of source matrix: each column gets value same as index
        src_index_np = np.tile( np.linspace( 0, float(ncols)-1, ncols ), nrows )
        src_index_np = src_index_np.reshape( (nrows, ncols) ).transpose( (1,0) )
        cls.src_index_t  = torch.from_numpy( src_index_np ).float()
        #print "src_index_np: ",self.src_index_np.shape#, self.src_index_np[3,:]

        cls.imgdimset = True

    @classmethod
    def print_intersect_grad(cls):
        print "Y2U: dy/du -------------- "
        w = 500
        for u in xrange(300,310):
            print " (w=500,u={}) ".format(u),cls.intersections_t[0,0,500,u+1]-cls.intersections_t[0,0,500,u]
        print "Y2U: dz/du -------------- "
        for u in xrange(300,310):
            print " (w=500,u={}) ".format(u),cls.intersections_t[0,1,500,u+1]-cls.intersections_t[0,1,500,u]
        print "Y2V: dy/dv -------------- "
        for v in xrange(300,310):
            print " (w=500,v={}) ".format(v),cls.intersections_t[1,0,500,v+1]-cls.intersections_t[1,0,500,v]
        print "Y2V: dz/dv -------------- "
        for v in xrange(300,310):
            print " (w=500,v={}) ".format(v),cls.intersections_t[1,1,500,v+1]-cls.intersections_t[1,1,500,v]


    @staticmethod
    def forward(ctx,pred_flowy2u, pred_flowy2v, source_originx, targetu_originx, targetv_originx  ):
        assert(IntersectUB.dataloaded and IntersectUB.imgdimset and IntersectUB.larcv_version is not None)

        ## our device
        dev = pred_flowy2u.device

        ## switch tensors to device
        IntersectUB.src_index_t = IntersectUB.src_index_t.to(device=dev)
        IntersectUB.intersections_t = IntersectUB.intersections_t.to(device=dev)
        #print pred_flowy2u.is_cuda
        #print IntersectUB.src_index_t.is_cuda
        #print IntersectUB.intersections_t.is_cuda

        ## img dims
        ncols = IntersectUB.ncols
        nrows = IntersectUB.nrows
        ntarget_wires = IntersectUB.ntarget_wires
        batchsize = pred_flowy2u.size()[0]

        if type(source_originx) is float:
            source_originx_t = torch.ones( (batchsize), dtype=torch.float ).to(device=dev)*source_originx
        else:
            source_originx_t = source_originx

        if type(targetu_originx) is float:
            targetu_originx_t = torch.ones( (batchsize), dtype=torch.float ).to(device=dev)*targetu_originx
        else:
            targetu_originx_t = targetu_originx

        if type(targetv_originx) is float:
            targetv_originx_t = torch.ones( (batchsize), dtype=torch.float ).to(device=dev)*targetv_originx
        else:
            targetv_originx_t = targetv_originx
        #print "source origin:  ",source_originx_t
        #print "targetu origin: ",targetu_originx_t
        #print "targetv origin: ",targetv_originx_t

        ## wire position calcs
        source_fwire_t       = torch.zeros( (batchsize,1,ncols,nrows), dtype=torch.float ).to( device=dev )
        pred_target1_fwire_t = torch.zeros( (batchsize,1,ncols,nrows), dtype=torch.float ).to( device=dev )
        pred_target2_fwire_t = torch.zeros( (batchsize,1,ncols,nrows), dtype=torch.float ).to( device=dev )
        for b in xrange(batchsize):

            ## we need to get the source wire, add origin wire + relative position
            source_fwire_t[b,:] = IntersectUB.src_index_t.add( source_originx_t[b] )

            ## calcualte the wires in the target planes
            pred_target1_fwire_t[b,:] = (IntersectUB.src_index_t+pred_flowy2u[b,:]).add( targetu_originx_t[b] )
            pred_target2_fwire_t[b,:] = (IntersectUB.src_index_t+pred_flowy2v[b,:]).add( targetv_originx_t[b] )

        ## clamp for those out of flow and round
        pred_target1_fwire_t.clamp(0,ntarget_wires).round()
        pred_target2_fwire_t.clamp(0,ntarget_wires).round()
        #print "source  fwire: ",source_fwire_t
        #print "target1 fwire: ",pred_target1_fwire_t
        #print "target2 fwire: ",pred_target2_fwire_t

        ## calculate the index for the lookup table
        pred_target1_index_t = (source_fwire_t*ntarget_wires + pred_target1_fwire_t).long()
        pred_target2_index_t = (source_fwire_t*ntarget_wires + pred_target2_fwire_t).long()

        ## get the (y,z) of the intersection we've flowed to
        posyz_target1_t = torch.zeros( (batchsize,2,ncols,nrows) ).to( device=dev )
        posyz_target2_t = torch.zeros( (batchsize,2,ncols,nrows) ).to( device=dev )
        for b in xrange(batchsize):
            posyz_target1_t[b,0,:,:] = torch.take( IntersectUB.intersections_t[0,0,:,:], pred_target1_index_t[b,0,:,:].reshape( ncols*nrows ) ).reshape( (ncols,nrows) ) # det-y
            posyz_target1_t[b,1,:,:] = torch.take( IntersectUB.intersections_t[0,1,:,:], pred_target1_index_t[b,0,:,:].reshape( ncols*nrows ) ).reshape( (ncols,nrows) ) # det-y
            posyz_target2_t[b,0,:,:] = torch.take( IntersectUB.intersections_t[1,0,:,:], pred_target2_index_t[b,0,:,:].reshape( ncols*nrows ) ).reshape( (ncols,nrows) ) # det-y
            posyz_target2_t[b,1,:,:] = torch.take( IntersectUB.intersections_t[1,1,:,:], pred_target2_index_t[b,0,:,:].reshape( ncols*nrows ) ).reshape( (ncols,nrows) ) # det-y

        #ctx.save_for_backward(posyz_target1_t,posyz_target2_t)
        #print "posyz_target1: ",posyz_target1_t
        #print "posyz_target2: ",posyz_target2_t

        return (posyz_target1_t,posyz_target2_t)

    @staticmethod
    def backward(ctx,grad_output1,grad_output2):
        #posyz_target1_t, posyz_target2_t, = ctx.saved_tensors
        #diffy = posyz_target1_t[0,:] - posyz_target2_t[0,:] # ydiff
        #diffz = posyz_target1_t[1,:] - posyz_target2_t[1,:] # zdiff
        batchsize = grad_output1.size()[0]
        grad_input_u = (-0.3464*grad_output1[:,0,:,:]).reshape( (batchsize,1,IntersectUB.ncols,IntersectUB.nrows) ) # only y-pos changes with respect to the intersection of Y-U wires
        grad_input_v = ( 0.3464*grad_output2[:,0,:,:]).reshape( (batchsize,1,IntersectUB.ncols,IntersectUB.nrows) ) # only y-pos changes with respect to the intersection of Y-V wires
        return grad_input_u,grad_input_v, None, None, None


if __name__=="__main__":

    device = torch.device("cuda:0")
    #device = torch.device("cpu")

    IntersectUB.load_intersection_data()
    IntersectUB.set_img_dims(512,832)
    IntersectUB.print_intersect_grad()

    # save a histogram
    rout = rt.TFile("testout_func_intersect_ub.root","recreate")
    ttest = rt.TTree("test","Consistency 3D Loss test data")
    dloss = array('d',[0])
    dtime = array('d',[0])
    ttest.Branch("loss",dloss,"loss/D")
    ttest.Branch("dtime",dtime,"dtime/D")

    # as test, we process some pre-cropped small samples
    io = larcv.IOManager()
    io.add_in_file( "../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root" ) # create a unit test file (csv)
    io.initialize()

    nentries = io.get_n_entries()
    print "Number of Entries: ",nentries
    start = time.time()

    istart=0
    iend=nentries
    #istart=155
    #iend=156

    for ientry in xrange(istart,iend):

        tentry = time.time()

        io.read_entry( ientry )
        if os.environ["LARCV_VERSION"]=="1":
            ev_adc_test = io.get_data(larcv.kProductImage2D,"adc")
            ev_flowy2u_test = io.get_data(larcv.kProductImage2D,"larflow_y2u")
            ev_flowy2v_test = io.get_data(larcv.kProductImage2D,"larflow_y2v")
            ev_trueflow_test = io.get_data(larcv.kProductImage2D,"pixflow")
            ev_truevisi_test = io.get_data(larcv.kProductImage2D,"pixvisi")
            flowy2u = ev_flowy2u_test.Image2DArray()[0]
            flowy2v = ev_flowy2v_test.Image2DArray()[0]
            truey2u = ev_trueflow_test.Image2DArray()[0]
            truey2v = ev_trueflow_test.Image2DArray()[1]
            visiy2u = ev_truevisi_test.Image2DArray()[0]
            visiy2v = ev_truevisi_test.Image2DArray()[1]
            source_meta  = ev_adc_test.Image2DArray()[2].meta()
            targetu_meta = ev_adc_test.Image2DArray()[0].meta()
            targetv_meta = ev_adc_test.Image2DArray()[1].meta()

        elif os.environ["LARCV_VERSION"]=="2":
            ev_adc_test = io.get_data("image2d","adc")
            ev_flowy2u_test = io.get_data("image2d","larflow_y2u")
            ev_flowy2v_test = io.get_data("image2d","larflow_y2v")
            ev_trueflow_test = io.get_data("image2d","pixflow")
            ev_truevisi_test = io.get_data("image2d","pixvisi")

            flowy2u = ev_flowy2u_test.as_vector()[0]
            flowy2v = ev_flowy2v_test.as_vector()[0]
            truey2u = ev_trueflow_test.as_vector()[0]
            truey2v = ev_trueflow_test.as_vector()[1]
            visiy2u = ev_truevisi_test.as_vector()[0]
            visiy2v = ev_truevisi_test.as_vector()[1]
            source_meta  = ev_adc_test.as_vector()[2].meta()
            targetu_meta = ev_adc_test.as_vector()[0].meta()
            targetv_meta = ev_adc_test.as_vector()[1].meta()


        # numpy arrays
        index = (0,1)
        if os.environ["LARCV_VERSION"]=="2":
            index = (1,0)

        np_flowy2u = larcv.as_ndarray(flowy2u).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_flowy2v = larcv.as_ndarray(flowy2v).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_visiy2u = larcv.as_ndarray(visiy2u).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_visiy2v = larcv.as_ndarray(visiy2v).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_trueflowy2u = larcv.as_ndarray(truey2u).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_trueflowy2v = larcv.as_ndarray(truey2v).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))

        #print "NAN indices (flow-Y2U): ",np.argwhere( np.isnan(np_flowy2u)  )
        #print "NAN indices (flow-Y2V): ",np.argwhere( np.isnan(np_flowy2v)  )
        #print "NAN indices (visi-Y2U): ",np.argwhere( np.isnan(np_visiy2u)  )
        #print "NAN indices (visi-Y2V): ",np.argwhere( np.isnan(np_visiy2v)  )

        # tensor conversion
        predflow_y2u_t = torch.from_numpy( np_flowy2u ).to(device=device).requires_grad_()
        predflow_y2v_t = torch.from_numpy( np_flowy2v ).to(device=device).requires_grad_()

        trueflow_y2u_t = torch.from_numpy( np_trueflowy2u ).to(device=device).requires_grad_()
        trueflow_y2v_t = torch.from_numpy( np_trueflowy2v ).to(device=device).requires_grad_()

        truevisi_y2u_t = torch.from_numpy( np_visiy2u ).to(device=device)
        truevisi_y2v_t = torch.from_numpy( np_visiy2v ).to(device=device)

        #print "requires grad: ",predflow_y2u_t.requires_grad,predflow_y2v_t.requires_grad
        #y2u_t = predflow_y2u_t
        #y2v_t = predflow_y2v_t
        y2u_t = trueflow_y2u_t
        y2v_t = trueflow_y2v_t

        source_origin = torch.zeros( (1) ).to(device=device)
        targetu_origin = torch.zeros( (1) ).to(device=device)
        targetv_origin = torch.zeros( (1) ).to(device=device)
        for b in xrange(1):
            source_origin[0]  = source_meta.min_x()
            targetu_origin[0] = targetu_meta.min_x()
            targetv_origin[0] = targetv_meta.min_x()

        posyz_fromy2u,posyz_fromy2v = IntersectUB.apply( y2u_t, y2v_t, source_origin, targetu_origin, targetv_origin )

        mask = truevisi_y2u_t*truevisi_y2v_t
        diff = (posyz_fromy2u-posyz_fromy2v)
        #print "diff.shape=",diff.shape
        #print "mask.shape=",mask.shape
        diff[:,0,:,:] *= mask[:,0,:,:]
        diff[:,1,:,:] *= mask[:,0,:,:]
        l2 = diff[:,0,:,:]*diff[:,0,:,:] + diff[:,1,:,:]*diff[:,1,:,:]
        #print "l2 shape: ",l2.shape

        if mask.sum()>0:
            lossval = l2.sum()/mask.sum()
        else:
            lossval = l2.sum()

        # backward test
        tback = time.time()
        lossval.backward()
        print "  runbackward: ",time.time()-tback," secs"

        print "Loss (iter {}): {}".format(ientry,lossval.item())," iscuda",lossval.is_cuda
        dloss[0] = lossval.item()
        dtime[0] = time.time()-tentry

        ttest.Fill()

    end = time.time()
    tloss = end-start
    print "Time: ",tloss," secs / ",tloss/nentries," secs per event"
    rout.cd()
    ttest.Write()
    rout.Close()
