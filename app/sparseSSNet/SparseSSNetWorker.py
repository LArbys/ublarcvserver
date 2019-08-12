#extra futures
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Base Imports
import os,sys,logging, zlib
import numpy as np
from larcv import larcv
from ctypes import c_int
from ublarcvserver import MDPyWorkerBase, Broker, Client
larcv.json.load_jsonutils()
from ROOT import std

import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable

# sparse uresnet imports (in networks/sparse_ssnet)
import uresnet
from uresnet.flags      import URESNET_FLAGS
from uresnet.main_funcs import inference
from uresnet.trainval   import trainval

"""
Implements worker for SLAC's sparse uresnet
"""

class SparseSSNetWorker(MDPyWorkerBase):

    # def __init__(self,broker_address,plane,
    #              weight_file,device,batch_size,
    #              use_half=False,
    #              **kwargs):
    def __init__(self,broker_address,plane,
                 weight_file,batch_size,
                 device_id=None,
                 use_half=False,
                 use_compression=False,
                 **kwargs):
        """
        Constructor

        inputs
        ------
        broker_address str IP address of broker
        plane int Plane ID number. Currently [0,1,2] only
        weight_file str path to files with weights
        batch_size int number of batches to process in one pass
        """
        if type(plane) is not int:
            raise ValueError("'plane' argument must be integer for plane ID")
        elif plane not in [0,1,2]:
            raise ValueError("unrecognized plane argument. \
                                should be either one of [0,1,2]")
        else:
            pass

        if type(batch_size) is not int or batch_size<0:
            raise ValueError("'batch_size' must be a positive integer")

        self.plane = plane
        self.batch_size = batch_size
        self._still_processing_msg = False
        self._use_half = use_half
        self._use_compression = use_compression
        if self._use_half:
            print("Using half of mrcnn not tested")
            assert 1==2
        service_name = "sparse_uresnet_plane%d"%(self.plane)
        super(SparseSSNetWorker,self).__init__( service_name,
                                                broker_address, **kwargs)

        # Get Configs going:
        # configuration from Ran:
        """
        inference --full -pl 1 -mp PATH_TO_Plane1Weights-13999.ckpt -io larcv_sparse 
              -bs 64 -nc 5 -rs 1 -ss 512 -dd 2 -uns 5 -dkeys wire,label 
              -mn uresnet_sparse -it 10 -ld log/ -if PATH_TO_INPUT_ROOT_FILE
        """
        self.config = uresnet.flags.URESNET_FLAGS()
        args = { "full":True,               # --full
                 "plane":self.plane,        # -pl
                 "model_path":weight_file,  # -mp
                 "io_type":"larcv_sparse",  # -io
                 "batch_size":1,            # -bs
                 "num_class":5,             # -nc
                 "report_step":1,           # -rs
                 "spatial_size":512,        # -ss
                 "data_dim":2,              # -dd
                 "uresnet_num_strides": 5,  # -uns
                 "data_keys":"wire,label",  # -dkeys
                 "model_name":"uresnet_sparse", # -mn
                 "iteration":10,            # -it
                 "log_dir":"log/",          # -ld
                 "input_file":"none" }      # -if
        self.config.update(args)
        
        print("\n\n-- CONFIG --")
        for name in vars(self.config):
            attribute = getattr(self.config,name)
            if type(attribute) == type(self.config.parser): continue
            print("%s = %r" % (name, getattr(self.config, name)))

        # Set random seed for reproducibility
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)


        self.trainval = trainval(self.config)
        self.trainval.initialize()
        self._log.info("Loaded ubMRCNN model. Service={}".format(service_name))

        # run random data (to test/reserve memory)
        N = 20000
        fake_sparse = np.zeros( (N,4) ) # (x,y,pixval,batch)
        fake_sparse[:,0] = np.random.randint( 0, 512, N ).astype(np.float)
        fake_sparse[:,1] = np.random.randint( 0, 512, N ).astype(np.float)
        fake_sparse[:,2] = np.random.random( [N] ) # coords and
        fake_sparse[:,3] = 0
        print("passing in fake data: ",fake_sparse.shape)
        data_blob = { 'data': [[fake_sparse]] }
        results = self.trainval.forward( data_blob )
        print("result keys: ",results.keys())


    def make_reply(self,request,nreplies):
        """we load each image and pass it through the net.
        we run one batch before sending off partial reply.
        the attribute self._still_processing_msg is used to tell us if we
        are still in the middle of a reply.
        """
        #print("DummyPyWorker. Sending client message back")
        self._log.debug("received message with {} parts".format(len(request)))

        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        # message pattern: [image_bson,image_bson,...]

        nmsgs = len(request)
        nbatches = nmsgs/self.batch_size

        if not self._still_processing_msg:
            self._next_msg_id = 0

        # collect messages for batch
        img2d_v  = []
        frames_used = []
        rseid_v = []
        npts_v  = []
        totpts  = 0        
        for imsg in xrange(self._next_msg_id,nmsgs):
            try:
                compressed_data = bytes(request[imsg])
                if self._use_compression:                
                    data = zlib.decompress(compressed_data)
                else:
                    data = compressed_data
                c_run = c_int()
                c_subrun = c_int()
                c_event = c_int()
                c_id = c_int()
                img2d = larcv.json.sparseimg_from_bson_pybytes( data, c_run, c_subrun, c_event, c_id )
            except:
                self._log.error("Image Data in message part {}\
                                could not be converted".format(imsg))
                continue
            self._log.debug("SparseImage[{}] converted: ".format(imsg))
            self._log.debug("   len(pixellist)={}".format(img2d.pixellist().size()))
            self._log.debug("   meta={}".format(img2d.meta(0).dump()))

            # check if correct plane!
            if img2d.meta(0).plane()!=self.plane:
                self._log.debug("Image[{}] is the wrong plane!".format(imsg))
                continue
            print("Worker has message, working...")

            npts = int(img2d.pixellist().size()/(img2d.nfeatures()+2))
            totpts += npts
            
            img2d_v.append(img2d)
            frames_used.append(imsg)
            npts_v.append(npts)
            rseid_v.append((c_run.value,c_subrun.value,c_event.value,c_id.value))
            if len(img2d_v)>=self.batch_size:
                self._next_msg_id = imsg+1
                break

        nimgs = len(img2d_v)
        self._log.debug("converted msgs into batch of {} images. frames={}"
                        .format(nimgs,frames_used))

        if nimgs==0:
            # send final message: ERROR
            return ["ERROR:nomessages".encode('utf-8')],True

        # form ndarray
        batch_np = np.zeros( ( totpts, 4 ) )
        startidx = 0
        idx      = 0
        for npts,img2d in zip( npts_v, img2d_v ):
            endidx   = startidx+npts
            print("img2d type: {}".format(img2d))
            spimg_np = larcv.as_ndarray( img2d, larcv.msg.kNORMAL )
            print("spimg_np shape: {}".format( spimg_np.shape ))
            print("batch_np[startidx:endidx,0:3] shape: {}".format(batch_np[startidx:endidx,0:3].shape))
            #print("spimg_np: {}".format(spimg_np[:,0:2]))
            #batch_np[startidx:endidx,0:3] = spimg_np[:,:]
            batch_np[startidx:endidx,0] = spimg_np[:,1]
            batch_np[startidx:endidx,1] = spimg_np[:,0]
            batch_np[startidx:endidx,2] = spimg_np[:,2]                        
            batch_np[startidx:endidx,3]   = idx
            print("batch_np: {}".format(batch_np[:,0:2]))
            idx += 1

        # pass to network
        data_blob = { 'data': [[batch_np]] }
        results = self.trainval.forward( data_blob )

        # format the resuls: store into sparseimage object
        print("results[softmax]: {}".format(type(results['softmax'])))
        
        reply = []
        startidx = 0
        for idx in xrange(len(results['softmax'])):
            ssnetout_np = results['softmax'][idx]
            print("ssneout_np: {}".format(ssnetout_np.shape))
            rseid = rseid_v[idx]
            meta  = img2d_v[idx].meta(0)
            npts  = int( npts_v[idx] )
            endidx = startidx+npts
            print("numpoints for img[{}]: {}".format(idx,npts))
            ssnetout_wcoords = np.zeros( (ssnetout_np.shape[0],ssnetout_np.shape[1]+2), dtype=np.float32 )
            ssnetout_wcoords[:,0:2] = batch_np[startidx:endidx,0:2]
            ssnetout_wcoords[:,2:2+ssnetout_np.shape[1]] = ssnetout_np[:,:]
            startidx = endidx
            print("ssnetout_wcoords: {}".format(ssnetout_wcoords[:,0:2]))

            meta_v = std.vector("larcv::ImageMeta")()
            for i in xrange(5):
                meta_v.push_back(meta)
            
            ssnetout_spimg = larcv.sparseimg_from_ndarray( ssnetout_wcoords, meta_v, larcv.msg.kDEBUG )
            bson = larcv.json.as_bson_pybytes( ssnetout_spimg, rseid[0], rseid[1], rseid[2], rseid[3] )
                                          
            if self._use_compression:
                compressed = zlib.compress(bson)
            else:
                compressed = bson
            reply.append(compressed)

        print("next message id: {}".format(self._next_msg_id))
        
        if self._next_msg_id>=nmsgs:
            isfinal = True
            self._still_processing_msg = False
        else:
            isfinal = False
            self._still_processing_msg = True

        self._log.debug("formed reply with {} frames. isfinal={}"
                        .format(len(reply),isfinal))
        return reply,isfinal

