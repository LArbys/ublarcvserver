from __future__ import print_function
import os,sys
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
                 cropper_cfg=None,
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

        self._input_mode      = input_mode
        self._use_compression = use_compression
        self._cropper_cfg     = cropper_cfg

        if self._input_mode == SparseSSNetClient.SPLIT:
            # SPLIT WHOLEVIEW IMAGE

            if cropper_cfg==None:
                # create config
                default_cfg="""ProcessDriver: {
  Verbosity: 0
  RandomAccess: false
  EnableFilter: false
  InputFiles: [""]
  IOManager: {
    IOMode: 2
    Name: "larflowinput"
    OutFileName: "tmp_out.root"
  }
  ProcessName: ["ubsplit"]
  ProcessType: ["UBSplitDetector"]

  ProcessList: {
    ubsplit: {
      Verbosity: 0
      InputProducer:\"%s\"
      OutputBBox2DProducer: \"detsplit\"
      CropInModule: true
      OutputCroppedProducer: \"detsplit\"
      BBoxPixelHeight: 512
      BBoxPixelWidth:  832
      CoveredZWidth: 310
      FillCroppedYImageCompletely: true
      DebugImage: false
      MaxImageS: -1
      RandomizeCrops: false
      MaxRandomAttempts: 1
      MinFracPixelsInCrop: 0.0
   }
 }
}
"""%( adc_producer )
                print(default_cfg,file=open("default_ubsplit.cfg",'w'))
                cropper_cfg = "default_ubsplit.cfg"
            
            self._splitter = larcv.ProcessDriver( "ProcessDriver" )
            self._splitter.configure( cropper_cfg )
            infiles = std.vector("std::string")()
            infiles.push_back( larcv_supera_file )
            self._splitter.override_input_file(infiles)
            self._splitter.initialize()
            self._inlarcv = self._splitter.io_mutable()
        else:
            self._inlarcv = larcv.IOManager(larcv.IOManager.kREAD,"",tick_direction)
            self._inlarcv.add_in_file(larcv_supera_file)
            self._inlarcv.initialize()

        # LARLITE INPUT (used when cropping based on OPFLASH)
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
        self._sparseout_tree_name = sparseout_tree_name
        self._adc_producer        = adc_producer
        self._opflash_producer    = opflash_producer

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

        if self._input_mode == SparseSSNetClient.SPLIT:
            
            # we split the entire image
            # hack with numpy for now

            self._splitter.process_entry( int(entry_num), False, False )
            ev_cropped = self._splitter.io_mutable().get_data(larcv.kProductImage2D, "detsplit")
            cropped_v = ev_cropped.Image2DArray()
            self._log.debug("cropped wholeview image into {} crop sets (total: {})".format(cropped_v.size()/3,cropped_v.size()))

            nsets = cropped_v.size()/3
            for iset in xrange(nsets):
                for p in xrange(3):
                    img = cropped_v.at( 3*iset+p )
                    img2d_v[p].append(img)
                

        elif self._input_mode == SparseSSNetClient.OPFLASH_ROI:
            # use the intime flash to look for a CROI
            # note, need to get data from larcv first else won't sync properly
            # this is weird behavior by larcv that I need to fix

            roi_v = []

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
            thresh_v  = std.vector("float")(1,0.5)
            require_v = std.vector("int")(1,1)
            for plane in range(nplanes):
                
                img = wholeview_v.at(plane)
                img2d_v[plane] = [img]

        else:
            raise ValueError("input mode [{}] not recognized".format(self._input_mode))


        # send messages and recieve replies
        replies = self.send_image_list(img2d_v,run=run,subrun=subrun,event=event)
        #print()
        #print("len(replies)", len(replies))
        #print("len(replies[0])", len(replies[0]))
        #print()
        
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
        #print("PLANES:", planes)
        for p in planes:
            #print()
            #print("plane: ", p)
            if p not in masks_v:
                masks_v[p] = []

            #if p not in [0]:
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
                self._log.debug("Number of replies: {}".format(len(workerout)))
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
            #print(imageid_received)
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

    def process_received_images(self, wholeview_v, replies_vv):
        """ receive the list of images from the worker 
        we create a track and shower image. 
        we also save the raw 5-class sparse image as well: HIP MIP SHOWER DELTA MICHEL
        """

        nplanes = wholeview_v.size()

        # save the sparse image data
        for p in xrange(nplanes):
            ev_sparse_output = self._outlarcv.get_data(larcv.kProductSparseImage,
                                                       "{}_plane{}".format(self._sparseout_tree_name,p) )

            replies_v = replies_vv[p]

            for reply in replies_v:
                ev_sparse_output.Append( reply )

            self._log.info("Saving {} sparse images for plane {}".format(ev_sparse_output.SparseImageArray().size(),p))

        # make the track/shower images
        #ev_shower = self._outlarcv.get_data(larcv.kProductImage2D, "sparseuresnet_shower" )
        #ev_track  = self._outlarcv.get_data(larcv.kProductImage2D, "sparseuresnet_track" )
        #ev_bg     = self._outlarcv.get_data(larcv.kProductImage2D, "sparseuresnet_background" )
        ev_pred   = self._outlarcv.get_data(larcv.kProductImage2D, "sparseuresnet_prediction" )

        # in order to seemlessly work with vertexer        
        ev_uburn = [ self._outlarcv.get_data(larcv.kProductImage2D, "uburn_plane{}".format(p) ) for p in xrange(3) ]
        
        #shower_v  = std.vector("larcv::Image2D")()
        #track_v   = std.vector("larcv::Image2D")()
        #bground_v = std.vector("larcv::Image2D")()
        pred_v    = std.vector("larcv::Image2D")()

        for p in xrange(wholeview_v.size()):
            img = wholeview_v.at(p)
            
            #showerimg = larcv.Image2D(img.meta())
            #showerimg.paint(0.0)
            #shower_v.push_back( showerimg )

            #trackimg  = larcv.Image2D(img.meta())
            #trackimg.paint(0.0)
            #track_v.push_back( trackimg )

            #bgimg  = larcv.Image2D(img.meta())
            #bgimg.paint(0.0)
            #bground_v.push_back( bgimg )

            predimg  = larcv.Image2D(img.meta())
            predimg.paint(0.0)
            pred_v.push_back( predimg )
            
        for p in xrange(wholeview_v.size()):

            predimg   = pred_v.at(p)
            wholemeta = wholeview_v.at(p).meta()

            uburn_track  = larcv.Image2D( wholeview_v.at(p).meta() )
            uburn_shower = larcv.Image2D( wholeview_v.at(p).meta() )
            uburn_track.paint(0.0)
            uburn_shower.paint(0.0)
            
            for sparseout in replies_vv[p]:
                npts = int( sparseout.pixellist().size()/(sparseout.nfeatures()+2) )
                #print("num points: {}".format(npts))
                #print("num features: {}".format(sparseout.nfeatures()+2))
                stride = int( sparseout.nfeatures()+2 )
                sparse_meta = sparseout.meta(0)
                
                for ipt in xrange(npts):

                    col = int(sparseout.pixellist().at( stride*ipt+0 ))
                    row = int(sparseout.pixellist().at( stride*ipt+1 ))
                    
                    hip = sparseout.pixellist().at( stride*ipt+2 )
                    mip = sparseout.pixellist().at( stride*ipt+3 )
                    shr = sparseout.pixellist().at( stride*ipt+4 )
                    dlt = sparseout.pixellist().at( stride*ipt+5 )
                    mic = sparseout.pixellist().at( stride*ipt+6 )

                    bg  = 1-(hip+mip+shr+dlt+mic)
                    totshr = shr+dlt+mic
                    tottrk = hip+mip

                    maxscore = 0.
                    maxarg   = 0
                    for i,val in enumerate([hip,mip,totshr]):
                        if val>maxscore:
                            maxarg = i
                            maxscore = val
                    if maxarg==1:
                        pred = 2 # track
                    elif maxarg==2:
                        pred = 3 # shower
                    else:
                        pred = 1 # hip

                    # translate to different meta
                    try:
                        xrow = wholemeta.row( sparse_meta.pos_y(row) )
                        xcol = wholemeta.col( sparse_meta.pos_x(col) )

                        #showerimg.set_pixel( xrow, xcol, totshr )
                        #trackimg.set_pixel(  xrow, xcol, tottrk )
                        #bgimg.set_pixel(     xrow, xcol, bg )
                        predimg.set_pixel(   xrow, xcol, pred )
                        uburn_track.set_pixel(  xrow, xcol, tottrk )
                        uburn_shower.set_pixel( xrow, xcol, totshr )
                    except:
                        self._log.info("error assigning {} -- {} to wholeview. meta={}".format( (col,row),
                                                                                                (sparse_meta.pos_x(col),
                                                                                                 sparse_meta.pos_y(row)),
                                                                                                sparse_meta.dump() ) )
            ev_uburn[p].Append( uburn_shower )
            ev_uburn[p].Append( uburn_track  )
            #ev_shower.Append( showerimg )
            #ev_track.Append(  trackimg )
            #ev_bg.Append( bgimg )
            ev_pred.Append( predimg )

        



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
