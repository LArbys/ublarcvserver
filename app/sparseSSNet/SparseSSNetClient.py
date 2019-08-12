import logging
from ublarcvserver import Client
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
import zlib
from ctypes import c_int, byref
from ROOT import std

larcv.json.load_jsonutils()

class SparseSSNetClient(Client):

    # Image process mode
    WHOLE=0
    SPLIT=1
    OPFLASH_ROI=2
    
    def __init__(self, broker_address,
                 larcv_supera_file,
                 output_larcv_filename,
                 adc_producer="wire",
                 input_mode=0,
                 larlite_opreco_file=None, opflash_producer="simpleFlashBeam",
                 tick_backwards=False,    sparseout_tree_name="uresnet",
                 use_compression=False,   use_sparseimg=True,
                 intimewin_min_tick=190,  intimewin_max_tick=320,
                 **kwargs):
        """
        broker_address        [str]   IP address and port of broker. e.g.: tcp://my.server.somwhere:6000
        larcv_supera_file     [str]   path to LArCV root file with whole images
        output_larcv_filename [str]   path to LArCV root where we will write output
        adc_producer          [str]   name of Tree containing input images. e.g. 'wire'
        skip_detsplit         [bool]  if true, process whole image at once. if false, process crops.
        opflash_producer      [str]   name of tree carrying opflash information (used to make CROI from flash) (deprecated)
        tick_backwards        [bool]  if true, expect input LArCV images to be stored in tick-backward format
        mrcnn_tree_name       [str]   name of output tree contaning MRCN
        use_compression       [bool]  if false (default), do not compress byte string sent and received
        use_sparseimg         [bool]  if false (default), do not convert whole image into sparse. otherwisek, do. 
                                      To save bytes transferred.
        intime_min_tick       [int]   Start of Time window  for trigger
        intime_max_tick       [int]   End of Time window  for trigger
        """
        super(SparseSSNetClient,self).__init__(broker_address,**kwargs)

        # setup the input and output larcv iomanager, input larlite manager
        tick_direction = larcv.IOManager.kTickForward
        if tick_backwards:
            tick_direction = larcv.IOManager.kTickBackward
        self._inlarcv = larcv.IOManager(larcv.IOManager.kREAD,"",tick_direction)
                                        
        self._inlarcv.add_in_file(larcv_supera_file)
        self._inlarcv.initialize()
        self._input_mode = input_mode
        self._use_compression = use_compression

        self._inlarlite = None
        if self._input_mode == SparseSSNetClient.OPFLASH_ROI:
            if larlite_opreco_file is None or os.path.exists(larlite_opreco_file):
                raise ValueError("larlite opreco file needed or not found when input mode is OPFLASH_ROI")
            self._inlarlite = LArliteManager(larlite.storage_manager.kREAD)
            self._inlarlite.add_in_filename(larlite_opreco_file)
            self._inlarlite.open()
            #self._inlarlite.set_verbosity(0)

        self._outlarcv = larcv.IOManager(larcv.IOManager.kWRITE)
        self._outlarcv.set_out_file(output_larcv_filename)
        self._outlarcv.set_verbosity(larcv.msg.kDEBUG)
        self._outlarcv.initialize()
        self._log = logging.getLogger(__name__)

        #FixedCROIFromFlash = ublarcvapp.ubdllee.FixedCROIFromFlashAlgo
        self._sparseout_tree_name  = sparseout_tree_name
        self._adc_producer         = adc_producer
        self._opflash_producer     = opflash_producer

        self._ubsplitdet = None

        # MESSAGES TO LOOKFOR
        self._ERROR_NOMSGS = "ERROR:nomessages".encode('utf-8')

    def get_entries(self):
        return self._inlarcv.get_n_entries()

    def __len__(self):
        return self.get_entries()

    def process_entry(self,entry_num):
        """ perform all actions -- send, receive, process, store -- for entry"""

        # get the entries
        ok = self._inlarcv.read_entry(entry_num)
        if not ok:
            raise RuntimeError("could not read larcv entry %d"%(entry_num))

        # get data
        ev_wholeview = self._inlarcv.get_data(larcv.kProductImage2D,self._adc_producer)
                                              
        wholeview_v = ev_wholeview.Image2DArray()
        nplanes = wholeview_v.size()

        run    = self._inlarcv.event_id().run()
        subrun = self._inlarcv.event_id().subrun()
        event  = self._inlarcv.event_id().event()
        self._log.info("num of planes in entry {}: {}".format((run,subrun,event),nplanes))

        # define the roi_v images
        img2d_v = {}
        for plane in range(nplanes):
            if plane not in img2d_v:
                img2d_v[plane] = []

        # for if we split
        roi_v = []
        
        if self._input_mode == SparseSSNetClient.SPLIT:
            # we split the entire image
            raise RuntimeError("Use of ubsplitdet for image not implemented")

        elif self._input_mode == SparseSSNetClient.OPFLASH_ROI:
            # use the intime flash to look for a CROI
            # note, need to get data from larcv first else won't sync properly
            # this is weird behavior by larcv that I need to fix
            self._inlarlite.syncEntry(self._inlarcv)
            ev_opflash = self._inlarlite.get_data(larlite.data.kOpFlash,
                                                  self._opflash_producer)
            nintime_flash = 0
            for iopflash in range(ev_opflash.size()):
                opflash = ev_opflash.at(iopflash)
                if ( opflash.Time()<self._intimewin_min_tick*0.015625
                     or opflash.Time()>self._intimewin_max_tick*0.015625):
                    continue
                flashrois=self._croi_fromflash_algo.findCROIfromFlash(opflash);
                for iroi in range(flashrois.size()):
                    roi_v.append( flashrois.at(iroi) )
                nintime_flash += 1
            self._log.info("number of intime flashes: ",nintime_flash)

            # make crops from the roi_v
            for roi in roi_v:
                for plane in range(nplanes):

                    wholeimg = wholeview_v.at(plane)
                    bbox = roi.BB(plane)
                    img2d = wholeimg.crop( bbox )

                    img2d_v[plane].append( img2d )

            planes = img2d_v.keys()
            sorted(planes)
            self._log.info("Number of images on each plane:")
            for plane in planes:
                self._log.info("  plane[{}]: {}".format(plane,len(img2d_v[plane])))

        elif self._input_mode==SparseSSNetClient.WHOLE:
            self._log.debug("Work on Full Images")
            thresh_v  = std.vector("float")(1,10.0)
            require_v = std.vector("int")(1,1)
            for plane in range(nplanes):
                
                img = wholeview_v.at(plane)
                img2d_v[plane] = [img]

                # pass sparse representation for effiency
                #tmp_v = std.vector("larcv::Image2D")()
                #tmp_v.push_back( img )
                #img2d_v[plane] = [larcv.SparseImage(tmp_v,thresh_v,require_v)]
        else:
            raise ValueError("input mode [{}] not recognized".format(self._input_mode))


        # send messages and recieve replies
        replies = self.send_image_list(img2d_v,run=run,subrun=subrun,event=event)
        print()
        print("len(replies)", len(replies))
        print("len(replies[0])", len(replies[0]))
        print()
        
        # format and store replies
        self.process_received_images(wholeview_v,replies)

        self._outlarcv.set_id( self._inlarcv.event_id().run(),
                               self._inlarcv.event_id().subrun(),
                               self._inlarcv.event_id().event())

        self._outlarcv.save_entry()
        return True


    def send_image_list(self,img2d_list, run=0, subrun=0, event=0):
        """ send all images in an event to the worker and receive msgs"""
        planes = img2d_list.keys()
        sorted(planes)
        rse = (run,subrun,event)
        self._log.info("sending images with rse={}".format(rse))

        thresholds = std.vector("float")(1,10.0)
        keep_pix   = std.vector("int")(1,1)

        masks_v = {}
        nsize_uncompressed = 0
        nsize_compressed = 0
        received_compressed = 0
        received_uncompressed = 0
        imageid_received = {}
        nimages_sent = 0
        print("PLANES:", planes)
        for p in planes:
            print()
            print("plane: ", p)
            if p not in masks_v:
                masks_v[p] = []

            #if p not in [2]:
            #    continue


            self._log.debug("sending images in plane[{}]: num={}."
                            .format(p,len(img2d_list[p])))

            # send image
            msg = []
            for img2d in img2d_list[p]:
                img_id = nimages_sent # make an id to track if it comes back

                # conversion into sparse image
                input_v = std.vector("larcv::Image2D")()
                input_v.push_back( img2d )

                spimg = larcv.SparseImage( input_v, thresholds, keep_pix )
                self._log.debug("sparse image made. size of pixellist(): {}".format(spimg.pixellist().size()))
                
                bson = larcv.json.as_bson_pybytes(spimg, run, subrun, event, img_id)
                                                  
                nsize_uncompressed += len(bson)
                if self._use_compression:
                    compressed = zlib.compress(bson)
                else:
                    compressed = bson
                nsize_compressed   += len(compressed)
                msg.append(compressed)
                # we make a flag to mark if we got this back
                imageid_received[img_id] = False
                nimages_sent += 1
            self.send(b"sparse_uresnet_plane%d"%(p),*msg)

            # receives
            isfinal = False
            while not isfinal:
                workerout = self.recv_part()
                isfinal =  workerout is None
                if isfinal:
                    self._log.debug("received done indicator by worker")
                    break
                self._log.debug("num frames received from worker: {}"
                                .format(len(workerout)))
                # use the time worker is preparing next part, to convert image
                print("Number of replies:", len(workerout))
                for reply in workerout:
                    if reply==self._ERROR_NOMSGS:
                        try:
                            if "ERROR" in reply.decode('utf-8'):
                                raise RuntimeError("Worker responds with ERROR: {}".format(reply))
                        except:
                            print("Could not decode byte object")
                            pass
                    data = bytes(reply)
                    received_compressed += len(data)
                    if self._use_compression:
                        data = zlib.decompress(data)
                    received_uncompressed += len(data)
                    c_run = c_int()
                    c_subrun = c_int()
                    c_event = c_int()
                    c_id = c_int()
                    replymask = larcv.json.sparseimg_from_bson_pybytes(data,
                                c_run, c_subrun, c_event, c_id )

                    rep_rse = (c_run.value,c_subrun.value,
                                c_event.value)
                    self._log.debug("rec images with rse={}".format(rep_rse))
                    if rep_rse!=rse:
                        self._log.warning("image with wronge rse={}. ignoring."
                                    .format(rep_rse))
                        continue
                    if c_id.value not in imageid_received:
                        self._log.warning("img id=%d not in IDset"%(c_id.value))
                        continue
                    else:
                        imageid_received[c_id.value] = True
                    self._log.debug("received image with correct rseid={}"
                                        .format(rep_rse))

                    masks_v[p].append(replymask)
                self._log.debug("running total, converted plane[{}] images: {}"
                                .format(p,len(masks_v[p])))
            complete = True
            # should be a multiple of 2: (shower,track)
            # if len(imgout_v[p])>0 and len(imgout_v[p])%2!=0:
            #     complete = False
            # should have got all images back
            print(imageid_received)
            for id,received in imageid_received.items():
                if not received:
                    complete = False
                    self._log.error("Did not receive image[%d]"%(id))

            if not complete:
                raise RuntimeError(\
                    "Did not receive complete set for all images")

        self._log.debug("Total sent size. uncompressed=%.2f MB compreseed=%.2f"\
                        %(nsize_uncompressed/1.0e6,nsize_compressed/1.0e6))
        self._log.debug("Total received. compressed=%.2f uncompressed=%.2f MB"\
                        %(received_compressed/1.0e6, received_uncompressed/1.0e6))
        return masks_v

    def process_received_images(self, wholeview_v, out_masks_vv):
        """ receive the list of images from the worker """

        if True:
            return
        
        nplanes = wholeview_v.size()
        # nplanes = 1
        # print("FORCING NPLANES=1")
        print(larcv.kProductClusterMask, type(larcv.kProductClusterMask))
        print(self._mrcnn_tree_name+"_masks", "{{type:", type(self._mrcnn_tree_name+"_masks"))
        ev_clustermasks = self._outlarcv.\
                        get_data(larcv.kProductClusterMask,
                                 self._mrcnn_tree_name+"_masks")
        print(len(ev_clustermasks.as_vector()))
        if len(ev_clustermasks.as_vector()) > 0:
            print(len(ev_clustermasks.as_vector()[0]))

        masks_vv = ev_clustermasks.as_vector()
        masks_vv.resize(nplanes)
        print("Survival?")
        # ev_track  = self._outlarcv.\
        #                 get_data(larcv.kProductImage2D,
        #                          self._mrcnn_tree_name+"_track")
        #ev_bg     = self._outlarcv.\
        #                get_data(larcv.kProductImage2D,"mrcnn_background")

        for p in range(nplanes):
            nmasks = int(len(out_masks_vv[p]))
            for mask in range(nmasks):
                print("Are we seeing probs?", out_masks_vv[p][mask].probability_of_class)
                masks_vv.at(p).push_back(out_masks_vv[p][mask])
        print(len(ev_clustermasks.as_vector()))
        if len(ev_clustermasks.as_vector()) > 0:
            print(len(ev_clustermasks.as_vector()[0]))



    def process_entries(self,start=0, end=-1):
        if end<0:
            end = self.get_entries()-1

        for ientry in range(start,end+1):
            self.process_entry(ientry)

    def finalize(self):
        self._inlarcv.finalize()
        self._outlarcv.finalize()
        if self._inlarlite != None:
            self._inlarlite.close()
