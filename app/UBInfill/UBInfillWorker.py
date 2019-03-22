import os,sys,logging, zlib
import numpy as np
from larcv import larcv
from ctypes import c_int
from ublarcvserver import MDPyWorkerBase, Broker, Client
larcv.json.load_jsonutils()

"""
Implements worker for Infill network
"""

class UBInfillWorker(MDPyWorkerBase):

    def __init__(self,broker_address,plane,
                 device,batch_size,
                 **kwargs):
        """
        Constructor

        inputs
        ------
        broker_address str IP address of broker
        plane int Plane ID number. Currently [0,1,2] only
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
        service_name = "infill_plane%d"%(self.plane)

        super(UBInfillWorker,self).__init__( service_name,
                                            broker_address, **kwargs)

        self.load_model(device)
        if self.is_model_loaded():
            self._log.info("Loaded ubInfill model. Service={}"\
                            .format(service_name))

    def load_model(self,device):
        # import pytorch
        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        # ----------------------------------------------------------------------
        # import model - change to my model
        if "INFILL_MODELDIR" in os.environ: # should have been placed there by configure.sh script
            INFILL_MODELDIR=os.environ["INFILL_MODELDIR"]
            sys.path.append(INFILL_MODELDIR)
        else:
            sys.path.append("/mnt/disk1/nutufts/kmason/ubdl/ublarcvserver/networks/infill")
        try:
            from ub_uresnet_infill import UResNetInfill
        except:
            raise RuntimeError("could not load Infill model. did you remember"
                            +" to setup?")

        if "cuda" not in device and "cpu" not in device:
            raise ValueError("invalid device name [{}]. Must str with name \
                                \"cpu\" or \"cuda:X\" where X=device number")

        self._log = logging.getLogger(self.idname())

        self.device = torch.device(device)

        self.model = UResNetInfill(inplanes=32,input_channels=1,num_classes=1,showsizes=True).to(self.device)
        self.model.eval()
        # ----------------------------------------------------------------------


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
                compressed_data = str(request[imsg])
                data = zlib.decompress(compressed_data)
                c_run = c_int()
                c_subrun = c_int()
                c_event = c_int()
                c_id = c_int()
                img2d = larcv.json.image2d_from_pystring(data,
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
        np_dtype = np.float32
        img_batch_np = np.zeros( (nimgs,1,sizes[0][0],sizes[0][1]),
                                    dtype=np_dtype )

        for iimg,img2d in enumerate(img2d_v):
            meta = img2d.meta()
            img2d_np = larcv.as_ndarray( img2d )\
                            .reshape( (1,1,meta.cols(),meta.rows()))

            img_batch_np[iimg,:] = img2d_np


        # now make into torch tensor
        img2d_batch_t = torch.from_numpy( img_batch_np ).to(self.device)
        out_batch_np = self.model(img2d_batch_t).detach().cpu().numpy()

        # compression techniques
        ## 1) threshold values to zero
        ## 2) suppress output for non-adc values
        ## 3) use half

        # suppress small values
        out_batch_np[ out_batch_np<1.0e-3 ] = 0.0

        # threshold
        for ich in xrange(out_batch_np.shape[1]):
            out_batch_np[:,ich,:,:][ img_batch_np[:,0,:,:]<10.0 ] = 0.0

        # convert back to full precision, if we used half-precision in the net

        self._log.debug("passed images through net. output batch shape={}"
                        .format(out_batch_np.shape))
        # convert from numpy array batch back to image2d and messages
        reply = []
        for iimg in xrange(out_batch_np.shape[0]):
            img2d = img2d_v[iimg]
            rseid = rseid_v[iimg]
            meta  = img2d.meta()
            for ich in xrange(out_batch_np.shape[1]):
                out_np = out_batch_np[iimg,ich,:,:]
                out_img2d = larcv.as_image2d_meta( out_np, meta )
                bson = larcv.json.as_pystring( out_img2d,
                                    rseid[0], rseid[1], rseid[2], rseid[3] )
                compressed = zlib.compress(bson)
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
