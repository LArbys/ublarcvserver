import os,sys,logging, zlib
import numpy as np
from larcv import larcv
from collections import OrderedDict

from ctypes import c_int
from ublarcvserver import MDPyWorkerBase, Broker, Client
larcv.json.load_jsonutils()

"""
Implements worker for Sparse Infill network
"""

class UBInfillSparseWorker(MDPyWorkerBase):

    def __init__(self,broker_address,plane,
                 weight_file,device,batch_size,
                 use_half=False,use_compression=False,
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
            print("PLANE GOOD: ", plane)
            pass

        if type(batch_size) is not int or batch_size<0:
            raise ValueError("'batch_size' must be a positive integer")

        self.plane = plane
        self.batch_size = batch_size
        self._still_processing_msg = False
        self._use_half = use_half
        self._use_compression = use_compression

        service_name = "infill_plane%d"%(self.plane)

        super(UBInfillSparseWorker,self).__init__( service_name,
                                            broker_address, **kwargs)

        self.load_model(weight_file,device,self._use_half)

        if self.is_model_loaded():
            self._log.info("Loaded ubInfill model. Service={}"\
                            .format(service_name))

    def load_model(self,weight_file,device,use_half):
        # import pytorch
        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        # ----------------------------------------------------------------------
        # import model - change to my model
        sys.path.append("../../../networks/infill")
        from sparseinfill import SparseInfill

        if "cuda" not in device and "cpu" not in device:
            raise ValueError("invalid device name [{}]. Must str with name \
                                \"cpu\" or \"cuda:X\" where X=device number")


        self._log = logging.getLogger(self.idname())

        self.device = torch.device(device)

        map_location = {"cuda:0":"cpu","cuda:1":"cpu"}
        self.model = SparseInfill( (512,496), 1,16,16,5, show_sizes=False)
        checkpoint = torch.load( weight_file, map_location=map_location )
        from_data_parallel = False
        for k,v in checkpoint["state_dict"].items():
            if "module." in k:
                from_data_parallel = True
                break

        if from_data_parallel:
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            checkpoint["state_dict"] = new_state_dict

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()


        print ("Loaded Model!")
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

        try:
            from ROOT import std
        except:
            raise RuntimeError("could not load ROOT.std")

        # message pattern: [image_bson,image_bson,...]

        nmsgs = len(request)
        nbatches = nmsgs/self.batch_size

        if not self._still_processing_msg:
            self._next_msg_id = 0

        # turn message pieces into numpy arrays
        imgdata_v  = []
        sizes    = []
        frames_used = []
        rseid_v = []
        totalpts =0
        for imsg in xrange(self._next_msg_id,nmsgs):
            try:
                compressed_data = str(request[imsg])
                if self._use_compression:
                    data = zlib.decompress(compressed_data)
                else:
                    data = compressed_data
                c_run = c_int()
                c_subrun = c_int()
                c_event = c_int()
                c_id = c_int()

                imgdata = larcv.json.sparseimg_from_bson_pybytes(data,
                                        c_run, c_subrun, c_event, c_id )
            except:
                self._log.error("Image Data in message part {}\
                                could not be converted".format(imsg))
                continue
            self._log.debug("Image[{}] converted: nfeatures={} npts={}"\
                            .format(imsg,imgdata.nfeatures(),
                                    imgdata.pixellist().size()/(imgdata.nfeatures()+1)))

            # get source meta
            # print ("nmeta=",imgdata.meta_v().size())
            srcmeta = imgdata.meta_v().front()
            # print( "srcmeta=",srcmeta.dump())

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
            totalpts += npts
            imgdata_v.append(imgdata)
            frames_used.append(imsg)
            rseid_v.append((c_run.value,c_subrun.value,c_event.value,c_id.value))
            if len(imgdata_v)>=self.batch_size:
                self._next_msg_id = imsg+1
                break


        # convert the images into numpy arrays
        nimgs = len(imgdata_v)
        self._log.debug("converted msgs into batch of {} images. frames={}"
                        .format(nimgs,frames_used))
        np_dtype = np.float32
        # img_batch_np = np.zeros( (nimgs,1,sizes[0][1],sizes[0][0]),
        #                             dtype=np_dtype )
        coord_np  = np.zeros( (totalpts,3), dtype=np.int )
        input_np = np.zeros( (totalpts,1), dtype=np_dtype )
        nfilled = 0
        for iimg,imgdata in enumerate(imgdata_v):
            (rows,cols,nfeatures,npts) = sizes[iimg]
            # print ("size of img", len(imgdata.pixellist()))

            if (len(imgdata.pixellist()) == 0):
                start = 0
                end = 1
                totalpts = end
                coord_np  = np.zeros( (totalpts,3), dtype=np.int )
                input_np = np.zeros( (totalpts,1), dtype=np_dtype )
                coord_np[start:end,0] = 0
                coord_np[start:end,1] = 0
                coord_np[start:end,2]   = iimg
                input_np[start:end,0]   = 10.1
                nfilled = 1

            else:
                data_np = larcv.as_ndarray( imgdata, larcv.msg.kNORMAL )
                start = nfilled
                end   = nfilled+npts
                coord_np[start:end,0:2] = data_np[:,0:2].astype(np.int)
                coord_np[start:end,2]   = iimg
                input_np[start:end,0]   = data_np[:,2]
                nfilled = end
            # print("shape of image: ",img2d_np.shape)

        coord_t  = torch.from_numpy( coord_np ).to(self.device)
        input_t = torch.from_numpy( input_np ).to(self.device)
        with torch.set_grad_enabled(False):
            out_t = self.model(coord_t, input_t, len(imgdata_v))

        out_t = out_t.detach().cpu().numpy()
        # convert from numpy array batch back to sparseimage and messages
        reply = []
        nfilled = 0
        for iimg,(imgshape,imgdata,rseid) in enumerate(zip(sizes,imgdata_v,rseid_v)):
            npts      = imgshape[3]
            start     = nfilled
            end       = start+npts
            nfeatures = 1
            # make numpy array to remake sparseimg
            sparse_np = np.zeros( (npts,2+nfeatures), dtype=np.float32 )
            sparse_np[:,0:2] = coord_np[start:end,0:2]
            sparse_np[:,2]   = out_t[start:end,0]

            outmeta_v = std.vector("larcv::ImageMeta")()
            outmeta_v.push_back( imgdata.meta_v().front() )

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
