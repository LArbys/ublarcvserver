import os,sys,time,logging,zlib
import numpy as np
from larcv import larcv
from ctypes import c_int
from ublarcvserver import MDPyWorkerBase, Broker, Client
from sparsemodels import load_models
larcv.json.load_jsonutils()

"""
Implements worker for Sparse LArFlow network
"""

class UBSparseLArFlowWorker(MDPyWorkerBase):

    def __init__(self,broker_address,plane,
                 weight_file,device,batch_size,
                 use_half=False,use_compression=False,
                 **kwargs):
        """
        Constructor

        inputs
        ------
        broker_address [str] IP address of broker
        plane [str or int] Plane from which we start the flow. Implemented choices ['Y']
        weight_file [str] path to files with weights
        batch_size [int] number of batches to process in one pass
        use_half [bool] if True, use half-precision (FP16) for forward pass
        """
        if type(plane) is not str and type(plane) is not int:
            raise ValueError("'plane' argument must be str or int for plane ID. e.g. 'Y' or 2")
        if type(plane) is str:
            if plane not in ['Y']:
                raise ValueError("unrecognized plane. current choices: 'Y'")
            planedict = {'Y':2}
            self.plane = planedict[plane]
        elif type(plane) is int:
            if plane not in [2]:
                raise ValueError("unrecognized plane. current choices: [2]")
            self.plane = plane


        if type(batch_size) is not int or batch_size<0:
            raise ValueError("'batch_size' must be a positive integer")

        self.batch_size = batch_size
        self._still_processing_msg = False
        self._use_half = use_half
        self._use_compression = use_compression
        service_name = "ublarflow_plane%d"%(self.plane)

        super(UBSparseLArFlowWorker,self).__init__( service_name,
                                                    broker_address, **kwargs)

        self.load_model(weight_file,device,self._use_half)
        if self.is_model_loaded():
            self._log.info("LOADED SPARSE-LARFLOW model. Service={} on device={}"\
                            .format(service_name,device))
        else:
            raise RuntimeError("ERROR LOADING UBSparseLArFlow Model")

    def load_model(self,weight_file,device,use_half):
        """
        weight_file [str] path to weights
        device [str] device name, e.g. "cuda:0" or "cpu"
        use_half [bool] if true, use half-precision (FP16)
        """

        # import pytorch
        try:
            import torch
            import sparseconvnet as scn
        except:
            raise RuntimeError("could not load pytorch!")

        # import model
        try:
            from sparselarflow import SparseLArFlow
        except Exception as e:
            raise RuntimeError("could not load SparseLArFlow model. did you remember"
                            +" to setup larflow (in the networks directory)?\n"
                            +"Exception: {}".format(e))

        if "cuda" not in device and "cpu" not in device:
            print ("cuda" not in device)
            raise ValueError("invalid device name ['{}' type={}]. Must str with name \
                                \"cpu\" or \"cuda:X\" where X=device number".format(device,type(device)))

        self._log = logging.getLogger(self.idname())

        self.device = torch.device(device)
        if not self._use_half:
            self.model = load_models("dualflow_v1",weight_file=weight_file).to(device=self.device)
        else:
            self.model = load_models("dualflow_v1",weight_file=weight_file).half().to(device=self.device)

        self.model.eval()

    def make_reply(self,request,nreplies):
        """we load each image and pass it through the net.
        we run one batch before sending off partial reply.
        the attribute self._still_processing_msg is used to tell us if we
        are still in the middle of a reply.
        """
        #print("DummyPyWorker. Sending client message back")
        self._log.debug("make_reply:: received message with {} parts".format(len(request)))

        if not self.is_model_loaded():
            self._log.debug("model not loaded for some reason. loading.")

        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        try:
            from ROOT import std
        except:
            raise RuntimeError("could not load ROOT.std")

        # message pattern: [image_bson,image_bson,...]

        nmsgs = len(request)
        nbatches = nmsgs/self.batch_size

        if not self._still_processing_msg:
            self._next_msg_id = 0

        # turn message pieces into larcv::SparseImage
        imgdata_v   = [] # actual sparse img
        sizes       = [] # tuples with (rows,cols,nfeatures,npoints)
        frames_used = []
        rseid_v     = []
        totalpts    = 0
        for imsg in xrange(self._next_msg_id,nmsgs):
            if self._use_compression:
                compressed_data = str(request[imsg])
                data = zlib.decompress(compressed_data)
            else:
                data = str(request[imsg])

            try:
                c_run = c_int()
                c_subrun = c_int()
                c_event = c_int()
                c_id = c_int()
                imgdata = larcv.json.sparseimg_from_bson_pybytes(data,
                                                                  c_run, c_subrun, c_event, c_id )
            except Exception as e:
                self._log.error("Image Data in message part {}".format(imsg)
                                +" could not be converted: {}".format(e))
                continue
            self._log.debug("Image[{}] converted: nfeatures={} npts={}"\
                            .format(imsg,imgdata.nfeatures(),
                                    imgdata.pixellist().size()/(imgdata.nfeatures()+2)))

            # get source meta
            print "nmeta=",imgdata.meta_v().size()
            srcmeta = imgdata.meta_v().at(0)
            print "srcmeta=",srcmeta.dump()
            # check if correct plane!
            if srcmeta.plane()!=self.plane:
                self._log.debug("Image[{}] meta plane ({}) is not same as worker's ({})!"
                                .format(imsg,srcmeta.plane(),self.plane))
                continue

            # check that same size as previous images
            nfeatures = imgdata.nfeatures()
            npts = imgdata.pixellist().size()/(2+nfeatures)
            imgsize = ( int(srcmeta.rows()), int(srcmeta.cols()),
                        int(nfeatures), int(npts) )
            if len(sizes)==0:
                sizes.append(imgsize)
            elif len(sizes)>0 and imgsize not in sizes:
                self._log.debug("Next image a different size. \
                                    we do not continue batch.")
                self._next_msg_id = imsg
                break
            # append information about image
            totalpts += npts
            imgdata_v.append(imgdata)
            frames_used.append(imsg)
            rseid_v.append((c_run.value,c_subrun.value,c_event.value,c_id.value))
            if len(imgdata_v)>=self.batch_size:
                self._next_msg_id = imsg+1
                break


        # convert the images into numpy arrays
        # need to make 4 arrays
        # coord_t (N,3) with N points of (row,col,batch)
        # srcpix_t (N,1) N points with adc values
        # tarpix_flow1_t (N,1) N points with adc values
        # tarpix_flow2_t (N,1) N points with adc values
        nimgs = len(imgdata_v)
        self._log.debug("converted msgs into batch of {} images. frames={}"
                        .format(nimgs,frames_used))
        np_dtype = np.float32
        if self._use_half:
            np_dtype = np.float16

        coord_np        = np.zeros( (totalpts,3), dtype=np.int )
        srcpix_np       = np.zeros( (totalpts,1), dtype=np_dtype )
        tarpix_flow1_np = np.zeros( (totalpts,1), dtype=np_dtype )
        tarpix_flow2_np = np.zeros( (totalpts,1), dtype=np_dtype )

        nfilled = 0
        for iimg,imgdata in enumerate(imgdata_v):
            (rows,cols,nfeatures,npts) = sizes[iimg]
            data_np = larcv.as_ndarray( imgdata, larcv.msg.kNORMAL )

            start = nfilled
            end   = nfilled+npts
            coord_np[start:end,0:2] = data_np[:,0:2].astype(np.int)
            coord_np[start:end,2]   = iimg

            if not self._use_half:
                srcpix_np[start:end,0]       = data_np[:,2]
                tarpix_flow1_np[start:end,0] = data_np[:,3]
                tarpix_flow2_np[start:end,0] = data_np[:,4]
            else:
                srcpix_np[start:end,0]       = data_np[:,2].astype(np.float16)
                tarpix_flow1_np[start:end,0] = data_np[:,3].astype(np.float16)
                tarpix_flow2_np[start:end,0] = data_np[:,4].astype(np.float16)
            nfilled = end

        # now make into torch tensors
        coord_t        = torch.from_numpy( coord_np ).to(self.device)
        srcpix_t       = torch.from_numpy( srcpix_np ).to(self.device)
        tarpix_flow1_t = torch.from_numpy( tarpix_flow1_np ).to(self.device)
        tarpix_flow2_t = torch.from_numpy( tarpix_flow2_np ).to(self.device)

        # run model: whole batch run at once
        tforward = time.time()
        with torch.set_grad_enabled(False):
            predict1_t, predict2_t = self.model(coord_t, srcpix_t,
                                                tarpix_flow1_t, tarpix_flow2_t,
                                                len(imgdata_v))

        out1_t = predict1_t.features.detach().cpu().numpy()
        out2_t = predict2_t.features.detach().cpu().numpy()
        dtforward = time.time()-tforward

        self._log.debug("passed images through net. input: {}. output batch shape={}. time={} secs"
                        .format(coord_t.shape,out1_t.shape,dtforward))

        # now need to make individual images

        # convert from numpy array batch back to sparseimage and messages
        reply = []
        nfilled = 0
        for iimg,(imgshape,imgdata,rseid) in enumerate(zip(sizes,imgdata_v,rseid_v)):
            npts      = imgshape[3]
            start     = nfilled
            end       = start+npts
            nfeatures = 2
            # make numpy array to remake sparseimg
            sparse_np        = np.zeros( (npts,2+nfeatures), dtype=np.float32 )
            sparse_np[:,0:2] = coord_np[start:end,0:2]
            sparse_np[:,2]   = out1_t[start:end,0]
            sparse_np[:,3]   = out2_t[start:end,0]

            outmeta_v = std.vector("larcv::ImageMeta")()
            outmeta_v.push_back( imgdata.meta_v().at(0) )
            outmeta_v.push_back( imgdata.meta_v().at(0) )

            # make the sparseimage object
            sparseimg = larcv.sparseimg_from_ndarray( sparse_np,
                                                      outmeta_v,
                                                      larcv.msg.kNORMAL )

            # convert to bson string
            bson = larcv.json.as_bson_pybytes( sparseimg,
                                                rseid[0], rseid[1], rseid[2], rseid[3] )
            # compress
            if self._use_compression:
                compressed = zlib.compress(bson)
            else:
                compressed = bson

            # add to reply message list
            reply.append(compressed)

            nfilled += npts

        if self._next_msg_id>=nmsgs:
            isfinal = True
            self._still_processing_msg = False
        else:
            isfinal = False
            self._still_processing_msg = True

        self._log.debug("formed reply with {} frames. isfinal={}"
                        .format(len(reply),isfinal))
        return reply,isfinal

    def is_model_loaded(self):
        return self.model is not None

if __name__ == "__main__":

    broker_address = "tcp://localhost:6009"
    plane='Y'
    weight_file="checkpoint.19400th.tar"
    device="cpu"
    batch_size=1
    worker=UBSparseLArFlowWorker(broker_address,plane,weight_file,
                                 device,batch_size)
    worker.connect()
    print "worker started: ",worker.idname()
    worker.run()

