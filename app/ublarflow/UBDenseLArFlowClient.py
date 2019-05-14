from __future__ import print_function
import logging, time
from ublarcvserver import Client
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from ROOT import std
import zlib
from ctypes import c_int, byref


larcv.json.load_jsonutils()

class UBDenseLArFlowClient(Client):

    def __init__(self, broker_address,
                 larcv_supera_file,
                 output_larcv_filename,
                 adc_producer="wire",
                 output_producer="larflow",
                 tick_backwards=False,
                 save_cropped_adc=False,
                 flow_dirs=["y2u","y2v"],
                 plane_scale_factors=[1.0,1.0,1.0],
                 **kwargs):
        """
        this class loads either larcv::sparseimage or larcv::image2d data from
        the input file, prepares the data into a binary json (bson) message to
        be sent to the broker. when the broker replies with worker output,
        save it to larcv root file.

        inputs
        ------
        broker_address str address of broker Socket
        larcv_supera_file str path to input data
        output_larcv_file str path to output file

        kwargs
        -------
        adc_producer str (default:"wire") name of ADC image2d tree
        output_producer str (default:"larflow") name of output flow info. will append flow_dir to name.
        tick_backwards bool (default:False) set to True if reading in LArCV1 files
        save_cropped_adc bool (default:False) save the ADC crops
        flow_dirs [list of str] direction of flow. options are ["y2u","y2v"]
        """
        super(UBDenseLArFlowClient,self).__init__(broker_address,**kwargs)

        # setup the input iomanager
        tick_direction = larcv.IOManager.kTickForward
        if tick_backwards:
            tick_direction = larcv.IOManager.kTickBackward
        self._inlarcv = larcv.IOManager(larcv.IOManager.kREAD,"",
                                        tick_direction)
        self._inlarcv.add_in_file(larcv_supera_file)
        self._inlarcv.initialize()

        # setup output iomanager
        self._outlarcv = larcv.IOManager(larcv.IOManager.kWRITE)
        self._outlarcv.set_out_file(output_larcv_filename)
        self._outlarcv.initialize()

        # setup config
        self._adc_producer    = adc_producer
        self._output_producer = output_producer

        # thresholds: adc values must be above this value to be included
        self._threshold_v   = std.vector("float")(3,10.0)

        # global scale factors to apply to ADC values for each plane
        self._plane_scale_factors = plane_scale_factors

        # setup logger
        self._log = logging.getLogger(__name__)

        # setup ubdetsplit
        fcfg = open("tmp_ubsplit.cfg",'w')
        print(self._get_default_ubsplit_cfg(),file=fcfg)
        fcfg.close()
        split_pset = larcv.CreatePSetFromFile( "tmp_ubsplit.cfg","UBSplitDetector")
        self._split_algo = ublarcvapp.UBSplitDetector()
        self._split_algo.configure(split_pset)
        self._split_algo.initialize()
        self._split_algo.set_verbosity(2)

        self.flow_dirs = flow_dirs

        # we do not stitch, but store individual crops
        # the post-processor does the stitching

    def _get_default_ubsplit_cfg(self):
        ubsplit_cfg="""
        InputProducer: \"wire\"
        OutputBBox2DProducer: \"detsplit\"
        CropInModule: true
        OutputCroppedProducer: \"detsplit\"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 832
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: -1
        RandomizeCrops: false
        MaxRandomAttempts: 1000
        MinFracPixelsInCrop: 0.0
        """
        return ubsplit_cfg

    def get_entries(self):
        return self._inlarcv.get_n_entries()

    def __len__(self):
        return self.get_entries()

    def process_entry(self,entry_num):
        """
        perform all actions -- send, receive, process, store -- for entry.

        we must:
        1) split full image into 3D-correlated crops for larflow
        2) package up into messages
        3) send, then get reply
        """

        # timing
        ttotal = time.time()

        tread = time.time()

        # get the entries
        ok = self._inlarcv.read_entry(entry_num)
        if not ok:
            raise RuntimeError("could not read larcv entry %d"%(entry_num))

        ev_wholeview = self._inlarcv.get_data(larcv.kProductImage2D,
                                              self._adc_producer)
        wholeview_v = ev_wholeview.Image2DArray()

        tread = time.time()-tread

        tsplit = time.time()
        roi_v  = std.vector("larcv::ROI")()
        crop_v = std.vector("larcv::Image2D")()
        self._split_algo.process( wholeview_v, crop_v, roi_v )
        tsplit = time.time()-tsplit
        print("split whole image into {} crops in {} secs".format(crop_v.size(),tsplit))

        flow_v = self.sendreceive_crops(crop_v,
                                        self._inlarcv.event_id().run(),
                                        self._inlarcv.event_id().subrun(),
                                        self._inlarcv.event_id().event())

        self.store_replies(flow_v, crop_v)

        self._outlarcv.set_id( self._inlarcv.event_id().run(),
                               self._inlarcv.event_id().subrun(),
                               self._inlarcv.event_id().event())

        self._outlarcv.save_entry()
        return True


    def sendreceive_crops(self,crop_v, run=0, subrun=0, event=0):
        """
        send all images in an event to the worker and receive msgs
        run, subrun, event ID used to make sure returning message is correct one

        the vector of crops is in sets of triplets:
        (plane0,plane1,plane2,plane0,plane1,plane2,...)

        we send doubles in order:
        ( source plane, target plane, source plane, target plane, ... )

        we expect to receive doublets for the flow:
        (flowy2u,flowy2v,flowy2u,flowy2uv,...)

        inputs
        ------
        crop_v vector<larcv::Image2D> cropped images
        run int run number to store in message
        subrun int subrun number to store in message
        event int event number to store in messsage
        """

        msg = {"y2u":[],"y2v":[]}

        thresh_v = std.vector("float")(1,10.0)
        rse = (run,subrun,event)
        got_reply = {"y2u":{},"y2v":{}}

        nimgsets = crop_v.size()/3

        for iset in xrange(nimgsets):
            srcimg = crop_v.at( 3*iset+2 )
            tary2u = crop_v.at( 3*iset+0 )
            tary2v = crop_v.at( 3*iset+1 )

            if self._plane_scale_factors[2]!=1.0:
                srcimg.scale_inplace( self._plane_scalefactors[2] )
            if self._plane_scale_factors[0]!=1.0:
                tary2u.scale_inplace( self._plane_scalefactors[0] )
            if self._plane_scale_factors[1]!=1.0:
                tary2v.scale_inplace( self._plane_scalefactors[1] )
                
            # to compress further, we represent as sparseimage
            src_v = std.vector("larcv::Image2D")()
            y2u_v = std.vector("larcv::Image2D")()
            y2v_v = std.vector("larcv::Image2D")()
            src_v.push_back(srcimg)
            y2u_v.push_back(tary2u)
            y2v_v.push_back(tary2v)

            src_sparse = larcv.SparseImage(src_v,thresh_v)
            y2u_sparse = larcv.SparseImage(y2u_v,thresh_v)
            y2v_sparse = larcv.SparseImage(y2v_v,thresh_v)

            bson_src = larcv.json.as_bson_pybytes(src_sparse,
                                          run, subrun, event, iset)
            bson_y2u = larcv.json.as_bson_pybytes(y2u_sparse,
                                          run, subrun, event, iset)
            bson_y2v = larcv.json.as_bson_pybytes(y2v_sparse,
                                          run, subrun, event, iset)

            nsize_uncompressed = 2*len(bson_src)+len(bson_y2u)+len(bson_y2v)
            compressed_src = zlib.compress(bson_src)
            compressed_y2u = zlib.compress(bson_y2u)
            compressed_y2v = zlib.compress(bson_y2v)
            nsize_compressed   = 2*len(compressed_src)\
                                + len(compressed_y2u) + len(compressed_y2v)

            msg["y2u"].append(compressed_src)
            msg["y2u"].append(compressed_y2u)
            msg["y2v"].append(compressed_src)
            msg["y2v"].append(compressed_y2v)

            got_reply["y2u"][iset] = 0
            got_reply["y2v"][iset] = 0
            print("produced msg[{}]: {}".format(iset,srcimg.meta().dump()))
            #if iset>=5:
            #    # truncate for debug
            #    break

        # we make a flag to mark if we got this back
        imageid_received = False
        imgout_v = {"y2u":{},"y2v":{}}
        received_compressed = 0
        received_uncompressed = 0
        complete = True
        for flowdir in self.flow_dirs:

            service_name = "ublarflow_dense_%s"%(flowdir)
            print("Send messages to service={}".format(service_name))
            self.send(service_name,*msg[flowdir])

            # receive this flow result
            isfinal = False
            while not isfinal:
                workerout = self.recv_part()
                isfinal =  workerout is None
                if isfinal:
                    self._log.debug("received DONE indicator by worker")
                    break
                self._log.debug("num frames received from worker: {}"
                                .format(len(workerout)))
                # use the time worker is preparing next part, to convert image
                for reply in workerout:
                    data = str(reply)
                    received_compressed += len(data)
                    data = zlib.decompress(data)
                    received_uncompressed += len(data)
                    c_run = c_int()
                    c_subrun = c_int()
                    c_event = c_int()
                    c_id = c_int()
                    replyimg = larcv.json.sparseimg_from_bson_pybytes(data,
                                c_run, c_subrun, c_event, c_id ).as_Image2D()
                    #byref(c_run),byref(c_subrun),byref(c_event),byref(c_id))
                    rep_rse = (c_run.value,c_subrun.value,c_event.value)
                    self._log.debug("rec images with rse={}".format(rep_rse))
                    if rep_rse!=rse:
                        self._log.warning("image with wronge rse={}. ignoring."
                                            .format(rep_rse))
                        continue

                    imgout_v[flowdir][int(c_id.value)] = replyimg
                    got_reply[flowdir][int(c_id.value)] += 1
                    self._log.debug("running total, converted images: {}"
                                    .format(len(imgout_v[flowdir])))

            # check completeness of this flow
            for imgid,nreplies in got_reply[flowdir].items():
                if nreplies!=1:
                    self._log.debug("imgid[{}] not complete. nreplies={}".format(imgid,nreplies))
                    complete = False
            self._log.debug("is {} compelete: {}".format(flowdir,complete))

        # should have got all images back
        if not complete:
            raise RuntimeError("Did not receive complete set for all images")

        self._log.debug("Total sent size. uncompressed=%.2f MB compreseed=%.2f"\
                        %(nsize_uncompressed/1.0e6,nsize_compressed/1.0e6))
        self._log.debug("Total received. compressed=%.2f uncompressed=%.2f MB"\
                        %(received_compressed/1.0e6, received_uncompressed/1.0e6))
        return imgout_v

    def store_replies(self, flowdict, crop_v):
        """ receive the list of images from the worker """
        for flowdir in ["y2u","y2v"]:
            evimg_output = self._outlarcv.\
                            get_data(larcv.kProductImage2D,
                                     self._output_producer+"_"+flowdir)

            # we only store adc images once, during the first flow
            if flowdir=="y2u":
                evadc_output = self._outlarcv.\
                               get_data(larcv.kProductImage2D,"adc")
            flowimgs = flowdict[flowdir]
            keys = flowimgs.keys()
            keys.sort()
            for iimg in keys:
                flowimg = flowimgs[iimg].front()
                evimg_output.Append(flowimg)
                if flowdir=="y2u":
                    # store adc images once during the first flow
                    for i in xrange(3):
                        evadc_output.Append( crop_v.at(3*iimg+i) )
                
        return True

    def process_entries(self,start=0, end=-1):
        if end<0:
            end = self.get_entries()-1

        for ientry in xrange(start,end+1):
            self.process_entry(ientry)

    def finalize(self):
        self._inlarcv.finalize()
        self._outlarcv.finalize()
