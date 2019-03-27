import os,sys,logging,zlib
import numpy as np
from larcv import larcv
from ctypes import c_int
from ublarcvserver import MDPyWorkerBase, Broker, Client
larcv.json.load_jsonutils()

"""
Implements worker for Sparse LArFlow network
"""

class UBDenseLArFlowWorker(MDPyWorkerBase):
    flow_dirs = ['y2u','y2v']

    def __init__(self,broker_address,flow_dir,
                 weight_file,device,batch_size,
                 use_half=False,
                 **kwargs):
        """
        Constructor

        inputs
        ------
        broker_address [str] IP address of broker
        flow_dir [str] Flow directions. Implemented choices: ['y2u','y2v']
        weight_file [str] path to files with weights
        device [str] device to run the network on
        batch_size [int] number of batches to process in one pass
        use_half [bool] if True, use FP16 for forward pass (default:False)
        """
        if type(flow_dir) is not str or flow_dir not in UBDenseLArFlowWorker.flow_dirs:
                raise ValueError("unrecognized flow_dir. current choices: {}"\
                                    .format(UBDenseLArFlowWorker.flow_dirs))
        self.flow_dir = flow_dir


        if type(batch_size) is not int or batch_size<0:
            raise ValueError("'batch_size' must be a positive integer")

        self.batch_size = batch_size
        self._still_processing_msg = False
        self._use_half = use_half
        service_name = "ublarflow_dense_{}".format(self.flow_dir)

        super(UBDenseLArFlowWorker,self).__init__( service_name,
                                                   broker_address, **kwargs)

        self.load_model(weight_file,device,self._use_half)
        if self.is_model_loaded():
            self._log.info("Loaded UBDenseLArFlowWorker model. Service={}"\
                            .format(service_name))

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
            from dense_larflow_funcs import load_dense_model
        except Exception as e:
            raise RuntimeError("could not load dense model. did you remember"
                            +" to setup larflow (in the networks directory)?\n"
                            +"Exception: {}".format(e))

        if "cuda" not in device and "cpu" not in device:
            raise ValueError("invalid device name [{}]. Must str with name \
                                \"cpu\" or \"cuda:X\" where X=device number")

        self._log = logging.getLogger(self.idname())

        self.device = torch.device(device)
        self.model = load_dense_model(weight_file,device=device,
                                        use_half=self._use_half ).to(self.device)

    def make_reply(self,request,nreplies):
        """
        we load each image and pass it through the net.
        we process all images before sending complete reply.
        the attribute self._still_processing_msg is used to tell us if we
        are still in the middle of a reply.
        """
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

        # turn message pieces into numpy arrays
        imgset_v = {}
        img2dset_v = {}
        rseid_v  = {}
        sizes    = [] # tuples with (rows,cols,nfeatures,npoints)
        frames_used = []
        totalpts = 0
        for imsg in xrange(self._next_msg_id,nmsgs):
            try:
                compressed_data = str(request[imsg])
                data = zlib.decompress(compressed_data)
                c_run = c_int()
                c_subrun = c_int()
                c_event = c_int()
                c_id = c_int()
                imgdata = larcv.json.sparseimg_from_bson_pystring(data,
                                        c_run, c_subrun, c_event, c_id )
            except Exception as e:
                self._log.error("Image Data in message part {}".format(imsg)
                                +" could not be converted: {}".format(e))
                continue
            self._log.debug("Image[{}] converted: nfeatures={} npts={}"\
                            .format(imsg,imgdata.nfeatures(),
                                    imgdata.pixellist().size()/(imgdata.nfeatures()+2)))
            #self._log.debug("Image[{}] meta: {}".format(imsg,imgdata.meta_v().front().dump()))

            # convert back to image2d
            imgid = c_id.value
            if imgid not in imgset_v:
                imgset_v[imgid] = []
                img2dset_v[imgid] = []
                rseid_v[imgid]=(c_run.value,c_subrun.value,c_event.value,imgid)
            img2d_v = imgdata.as_Image2D()
            print(img2d_v.front().meta().dump())
            imgset_v[imgid].append( img2d_v.front() )
            img2dset_v[imgid].append(img2d_v)

        # run the network and produce replies

        # we run in pairs of (src,target) crops. responsibility of Client
        # to get this correct
        keys = imgset_v.keys()
        keys.sort()
        nsets = len(keys)
        current_cols = 0
        current_rows = 0
        src_np = None
        tar_np = None
        ibatch = 0
        iset = 0
        flow_v = {}
        set_v = []
        meta_v = {}

        while iset<nsets:
            setid = keys[iset]
            if len(imgset_v[setid])!=2:
                # set is not complete
                iset += 1
                continue
            if imgset_v[setid][0].meta().plane()==2:
                src = imgset_v[setid][0]
                tar = imgset_v[setid][1]
            else:
                src = imgset_v[setid][1]
                tar = imgset_v[setid][0]
                imgset_v[setid] = []
                imgset_v[setid].append(src)
                imgset_v[setid].append(tar)

            # if first set of images, create numpy array
            if src_np is None:
                imgnptype = np.float32
                if self._use_half:
                    imgnptype = np.float16
                print("src_np: {}".format((self.batch_size,1,src.meta().rows(),src.meta().cols())))
                src_np = np.zeros( (self.batch_size,1,src.meta().rows(),src.meta().cols()), dtype=imgnptype )
                tar_np = np.zeros( (self.batch_size,1,tar.meta().rows(),tar.meta().cols()), dtype=imgnptype )
                set_v = []
                meta_v = {}

            # check that same size as previous images
            samesize = True
            if src_np.shape[2]!=src.meta().rows() or src_np.shape[3]!=src.meta().cols():
                samesize = False

            # if same size and we have not filled the batch yet, add to batch array
            if samesize and ibatch<self.batch_size:
                src_np[ibatch,0,:,:] = np.transpose( larcv.as_ndarray(src), (1,0) )
                tar_np[ibatch,0,:,:] = np.transpose( larcv.as_ndarray(tar), (1,0) )
                meta_v[setid] = src.meta()
                set_v.append(setid)
                iset += 1
                ibatch += 1

            if not samesize or ibatch==self.batch_size or iset==nsets:
                # convert to torch and run the batch through the network
                src_t  = torch.from_numpy(src_np).to(self.device)
                tar_t  = torch.from_numpy(tar_np).to(self.device)
                flow_t, visi_t = self.model( src_t, tar_t )

                # repress flow_t for values below threshold
                flow_t[ torch.lt(src_t,10.0) ] = 0.0

                # convert back to image2d. only use those with setid
                flow_np = flow_t.detach().cpu().numpy().astype(np.float32)
                for ib,sid in enumerate(set_v):
                    # convert back to image2d
                    flow_v[sid] = larcv.as_image2d_meta(flow_np[ib,0,:,:],meta_v[sid])

                # reset batch variables
                set_v = []
                ibatch = 0
                src_np = None
                tar_np = None

        # turn image2d into sparseimage and ship back to client
        reply = []
        isfinal = True
        nfilled = 0

        for setid in keys:

            flow = flow_v[setid]
            flowpix = larcv.as_pixelarray_with_selection( flow,
                                                          imgset_v[setid][0],
                                                          10.0 )
            # make the sparseimage object
            outmeta_v = std.vector("larcv::ImageMeta")()
            outmeta_v.push_back( imgset_v[setid][0].meta() )
            sparseimg = larcv.sparseimg_from_ndarray( flowpix,
                                                      outmeta_v,
                                                      larcv.msg.kNORMAL )

            # convert to bson string
            rseid = rseid_v[setid]
            bson = larcv.json.as_bson_pystring( sparseimg,
                                        rseid[0], rseid[1], rseid[2], rseid[3] )
            # compress
            compressed = zlib.compress(bson)

            # add to reply message list
            reply.append(compressed)


        self._log.debug("formed reply with {} frames. isfinal={}"
                        .format(len(reply),isfinal))
        return reply,isfinal

    def is_model_loaded(self):
        return self.model is not None
