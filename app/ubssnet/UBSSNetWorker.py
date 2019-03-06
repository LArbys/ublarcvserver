import os,sys,logging

from ublarcvserver import MDPyWorkerBase, Broker, Client

"""
Implements worker for ubssnet
"""

class UBSSNetWorker(MDPyWorkerBase):

    def __init__(self,broker_address,plane,
                weight_file,device,batch_size,
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
        service_name = "ubssnet_plane%d"%(self.plane)

        super(UBSSNetWorker,self).__init__( service_name,
                                            broker_address, **kwargs)

        self.load_model(weight_file,device)
        if self.is_model_loaded():
            self._log.info("Loaded ubSSNet model. Service={}"\
                            .format(service_name))

    def load_model(self,weight_file,device):
        # import pytorch
        try:
            import torch
        except:
            raise RuntimeError("could not load pytorch!")

        # import model
        try:
            from ubssnet import ubSSNet
        except:
            raise RuntimeError("could not load ubSSNet model. did you remember"
                            +" to setup pytorch-uresnet?")

        if "cuda" not in device and "cpu" not in device:
            raise ValueError("invalid device name [{}]. Must str with name \
                                \"cpu\" or \"cuda:X\" where X=device number")

        self._log = logging.getLogger(__name__)

        self.device = torch.device(device)
        self.model = ubSSNet(weight_file).to(self.device)
        self.model.eval()


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

        # message pattern: [image_bson,image_bson,...]

        nmsgs = len(request)
        nbatches = nmsgs/self.batch_size

        if not self._still_processing_msg:
            self._next_msg_id = 0

        # turn message pieces into numpy arrays
        img2d_v  = []
        sizes    = []
        for imsg in xrange(self._next_msg_id,nmsgs):
            try:
                img2d = larcv.json.image2d_from_pystring( str(request[imsg]) )
            except:
                self._log.error("Image Data in message part {}\
                                could not be converted".format(nreplies))
                continue
            self._log.debug("Image[{}] converted: {}"\
                            .format(imsg,img2d.meta().dump()))

            # check if correct plane!
            if img2d.meta().plane()!=self.plane:
                self._log.debug("Image[{}] is the wrong plane!".format(imsg))
                continue

            # check that same size as previous images
            imgsize = (img2d.meta().cols(),img2d.meta().rows)
            if len(sizes)==0:
                sizes.append(imgsize)
            elif len(sizes)>0 and imgsize not in sizes:
                self._log.debug("Next image a different size. \
                                    we do not continue batch.")
                self._next_msg_id = imsg+1
                break
            img2d_v.append(img2d)
            if len(img2d_v)>=self.batch_size:
                self._next_msg_id = imsg+1
                break


        # convert the images into numpy arrays
        img_batch_np = np.zeros( (len(img2d_v),1,sizes[0][0],sizes[0][1]),\
                                    dtype=np.float32 )

        for iimg,img2d in enumerate(img2d_v):
            meta = img2d.meta()
            img2d_np = larcv.as_ndarray( img2d )\
                            .reshape( (1,1,meta.cols(),meta.rows()))
            img_batch_np[iimg,:] = img2d_np

        # now make into torch tensor
        img2d_batch_t = torch.from_numpy( img_batch_np ).to(self.device)
        out_batch_np = self.model(img2d_batch_t).detach().numpy()

        # convert from numpy array batch back to image2d and messages
        reply = []
        for out_np,img2d in zip(out_batch_np,img2d_v):
            meta = img2d.meta()
            out_img2d = larcv.as_image2d_meta( out_np, meta )
            bson = larcv.json.as_pystring( out_img2d )
            reply.append(bson)

        if self._next_msg_id>=nreplies:
            isfinal = True
            self._still_processing_msg = False
        else:
            isfinal = False
            self._still_processing_msg = True

        return reply,isfinal

    def is_model_loaded(self):
        return self.model is not None
