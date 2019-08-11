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

import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable

# sparse uresnet imports (in networks/sparse_ssnet)
import uresnet
from uresnet.flags      import URESNET_FLAGS
from uresnet.main_funcs import inference
from uresnet.trainval   import trainval

#for LArCVDataset
#import os,time
#import ROOT
#from larcv import larcv
#import numpy as np
#from torch.utils.data import Dataset

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
        N = 10000
        fake_sparse = np.zeros( (N,4) ) # (x,y,pixval,batch)
        fake_sparse[:,0] = np.random.randint( 0, 512, N ).astype(np.float)
        fake_sparse[:,1] = np.random.randint( 0, 512, N ).astype(np.float)
        fake_sparse[:,2] = np.random.random( [N] ) # coords and
        fake_sparse[:,3] = 0
        print("passing in fake data: ",fake_sparse.shape)
        data_blob = { 'data': [[fake_sparse]] }
        results = self.trainval.forward( data_blob )
        print("result keys: ",results.keys())



    # def load_model(self,weight_file,device,use_half):
    def load_model(self,weight_file,use_half,device_id):

        # import pytorch
        # self._log.info("load_model does not use device, or use_half in MRCNN")
        # print("Device in load model: ", device)

        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        # import model
        try:
            from modeling.model_builder import Generalized_RCNN as ubMRCNN
            # from ubmrcnn import ubMRCNN
        except:
            raise RuntimeError("could not load ubMRCNN model. did you remember"
                            +" to setup everything?")

        # if "cuda" not in device and "cpu" not in device:
        #     raise ValueError("invalid device name [{}]. Must str with name \
        #                         \"cpu\" or \"cuda:X\" where X=device number")

        self._log = logging.getLogger(self.idname())

        # self.device = torch.device(device)

        self.model = Generalized_RCNN()

        #checkpoint = torch.load(weight_file, map_location=lambda storage, loc: storage)
        locations = {}
        for x in range(6):
            locations["cuda:%d"%(x)] = "cpu"
        checkpoint = torch.load(weight_file, map_location=locations)

        # self.device = device_id
        # if device_id==None:
        #     self.device = torch.device("cpu")
        # else:
        #     self.device = torch.device("cuda:%d"%(device_id))


        net_utils.load_ckpt(self.model, checkpoint['model'])
        # self.model = mynn.DataParallel(self.model, cpu_keywords=['im_info', 'roidb'],
        #                                minibatch=True, device_ids=[0],
        #                                output_device=0)  # only support single GPU

        self.model = mynn.DataSingular(self.model, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True , device_id=[cfg.MODEL.DEVICE])

        self.model.eval()
        #for x,t in self.model.state_dict().items():
        #    print(x,t.device)


    def make_reply(self,request,nreplies):
        """we load each image and pass it through the net.
        we run one batch before sending off partial reply.
        the attribute self._still_processing_msg is used to tell us if we
        are still in the middle of a reply.
        """
        #print("DummyPyWorker. Sending client message back")
        self._log.debug("received message with {} parts".format(len(request)))

        if not self.is_model_loaded():
            self._log.debug("model not loaded for some reason. loading.")

        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        # message pattern: [image_bson,image_bson,...]

        nmsgs = len(request)
        nbatches = nmsgs/self.batch_size

        if not self._still_processing_msg:
            self._next_msg_id = 0

        # turn message pieces into numpy arrays
        img2d_v  = []
        sizes    = []
        frames_used = []
        rseid_v = []
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
                img2d = larcv.json.image2d_from_pybytes(data,
                                        c_run, c_subrun, c_event, c_id )
            except:
                self._log.error("Image Data in message part {}\
                                could not be converted".format(imsg))
                continue
            self._log.debug("Image[{}] converted: {}"\
                            .format(imsg,img2d.meta().dump()))

            # check if correct plane!
            if img2d.meta().plane()!=self.plane:
                self._log.debug("Image[{}] is the wrong plane!".format(imsg))
                continue
            print("Worker Has message, working...")
            # check that same size as previous images
            imgsize = (int(img2d.meta().cols()),int(img2d.meta().rows()))
            if len(sizes)==0:
                sizes.append(imgsize)
            elif len(sizes)>0 and imgsize not in sizes:
                self._log.debug("Next image a different size. \
                                    we do not continue batch.")
                self._next_msg_id = imsg
                break
            img2d_v.append(img2d)
            frames_used.append(imsg)
            rseid_v.append((c_run.value,c_subrun.value,c_event.value,c_id.value))
            if len(img2d_v)>=self.batch_size:
                self._next_msg_id = imsg+1
                break


        # convert the images into numpy arrays
        nimgs = len(img2d_v)
        self._log.debug("converted msgs into batch of {} images. frames={}"
                        .format(nimgs,frames_used))

        if nimgs==0:
            # send final message: ERROR
            return ["ERROR:nomessages".encode('utf-8')],True

        np_dtype = np.float32
        if self._use_half:
            np_dtype = np.float16
        img_batch_np = np.zeros( (nimgs,1,sizes[0][0],sizes[0][1]),
                                    dtype=np_dtype )
        for iimg,img2d in enumerate(img2d_v):
            meta = img2d.meta()
            img2d_np = larcv.as_ndarray( img2d )\
                            .reshape( (1,1,meta.cols(),meta.rows()))
            if not self._use_half:
                img_batch_np[iimg,:] = img2d_np
            else:
                img_batch_np[iimg,:] = img2d_np.as_type(np.float16)
    #

        # print()
        # print('img_batch_np', img_batch_np.shape)
        # print()
        clustermasks_all_imgs = []
        mask_count = 0
        for img_num in range(img_batch_np.shape[0]):
            meta = img2d_v[img_num].meta()
            height = img_batch_np.shape[2]
            width = img_batch_np.shape[3]
            # Old and Slow:
            # im = np.zeros ((width,height,3))
            #
            # for h in range(img_batch_np.shape[2]):
            #     for w in range(img_batch_np.shape[3]):
            #         value = img_batch_np[img_num][0][h][w]
            #         im[w][h][:] = value
            #
            # print("im.shape", im.shape)
            # New and Fast
            im = np.array([np.copy(img_batch_np[0][0]),np.copy(img_batch_np[0][0]),np.copy(img_batch_np[0][0])])
            im = np.moveaxis(np.moveaxis(im,0,2),0,1)

            assert im is not None
            thresh = 0.7
            print("Using a score threshold of 0.7 to cut boxes. Hard Coded")
            clustermasks_this_img = []
            cls_boxes, cls_segms, cls_keyps, round_boxes = im_detect_all(self.model, im, timers=None, use_polygon=False)
            np.set_printoptions(suppress=True)
            for cls in range(len(cls_boxes)):
                assert len(cls_boxes[cls]) == len(cls_segms[cls])
                assert len(cls_boxes[cls]) == len(round_boxes[cls])
                for roi in range(len(cls_boxes[cls])):
                    if cls_boxes[cls][roi][4] > thresh:
                        segm_coo = cls_segms[cls][roi].tocoo()
                        non_zero_num = segm_coo.count_nonzero()
                        segm_np = np.zeros((non_zero_num, 2), dtype=np.float32)
                        counter = 0
                        for i,j,v in zip(segm_coo.row, segm_coo.col, segm_coo.data):
                            segm_np[counter][0] = j
                            segm_np[counter][1] = i
                            counter = counter+1
                        round_box = np.array(round_boxes[cls][roi], dtype=np.float32)
                        round_box = np.append(round_box, np.array([cls], dtype=np.float32))



                        clustermasks_this_img.append(larcv.as_clustermask(segm_np, round_box, meta, np.array([cls_boxes[cls][roi][4]], dtype=np.float32)))
                        mask_count = mask_count + 1
                        ### Checks to make sure the clustermasks being placed
                        ### in the list have the appropriate values relative
                        ### to what we send the pyutil
                        # cmask = clustermasks_this_img[len(list_clustermasks)-1]
                        # print()
                        # print(round_box)
                        # print("Segm shape", segm_np.shape)
                        # print(segm_np[0][0] , segm_np[0][1])
                        # print(segm_np[1][0] , segm_np[1][1])
                        # print()
                        # print(cmask.box.min_x(), cmask.box.min_y(), "   ", cmask.box.max_x(), cmask.box.max_y(), "    " , cmask.type)
                        # print(cmask._box.at(0), cmask._box.at(1), "   ", cmask._box.at(2), cmask._box.at(3), "    " , cmask._box.at(4))
                        # print("points_v len", len(cmask.points_v))
                        # print(cmask.points_v.at(0).x, cmask.points_v.at(0).y)
                        # print(cmask.points_v.at(1).x, cmask.points_v.at(1).y)
            clustermasks_all_imgs.append(clustermasks_this_img)
        print("Mask Count:", mask_count, " in plane: ", self.plane)
        # remove background values
        out_batch_np = img_batch_np
        # print("type(out_batch_np)", type(out_batch_np))
        out_batch_np = out_batch_np[:,1:,:,:]

        # compression techniques
        ## 1) threshold values to zero
        ## 2) suppress output for non-adc values
        ## 3) use half

        # # suppress small values
        # out_batch_np[ out_batch_np<1.0e-3 ] = 0.0
        #
        # # threshold
        # for ich in xrange(out_batch_np.shape[1]):
        #     out_batch_np[:,ich,:,:][ img_batch_np[:,0,:,:]<10.0 ] = 0.0

        # convert back to full precision, if we used half-precision in the net
        if self._use_half:
            out_batch_np = out_batch_np.as_type(np.float32)


        self._log.debug("passed images through net. Number of images = {} total masks across all images = {}"
                        .format(len(clustermasks_all_imgs), mask_count))
        # convert from numpy array batch back to image2d and messages
        # print("out_batch_np.shape",out_batch_np.shape)
        # out_batch_np = np.zeros((1,1,3456,1008),dtype=np.float32)
        reply = []
        for idx in xrange(len(clustermasks_all_imgs)):
            clustermask_set = clustermasks_all_imgs[idx]
            rseid = rseid_v[iimg]

            for mask_idx in xrange(len(clustermask_set)):
                mask = clustermask_set[mask_idx]
                # print(mask.as_vector_box_no_convert()[0], mask.as_vector_box_no_convert()[1], mask.as_vector_box_no_convert()[2], mask.as_vector_box_no_convert()[3])
                meta  = mask.meta
                # print((meta.dump()))
                # print(type(mask))
                # out_img2d = larcv.as_image2d_meta( out_np.reshape((1008,3456)), meta )
                bson = larcv.json.as_pybytes( mask,
                                    rseid[0], rseid[1], rseid[2], rseid[3] )
                if self._use_compression:
                    compressed = zlib.compress(bson)
                else:
                    compressed = bson
                reply.append(compressed)
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
