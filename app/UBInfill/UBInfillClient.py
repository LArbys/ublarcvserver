from __future__ import print_function
import numpy

import logging
from ublarcvserver import Client
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
import zlib
from ctypes import c_int, byref


#larcv.load_pyutils()
larcv.json.load_jsonutils()

class UBInfillClient(Client):

    def __init__(self, broker_address,
                    larcv_supera_file,
                    output_larcv_filename,
                    adc_producer="wire", chstatus_producer="wire",
                    tick_backwards=True, infill_tree_name="infill",**kwargs):
        """
        """
        super(UBInfillClient,self).__init__(broker_address,**kwargs)

        # setup the input and output larcv iomanager, input larlite manager
        tick_direction = larcv.IOManager.kTickForward
        if tick_backwards:
            tick_direction = larcv.IOManager.kTickBackward
        self._inlarcv = larcv.IOManager(larcv.IOManager.kREAD,"",
                                        tick_direction)
        self._inlarcv.add_in_file(larcv_supera_file)
        self._inlarcv.initialize()

        self._outlarcv = larcv.IOManager(larcv.IOManager.kWRITE)
        self._outlarcv.set_out_file(output_larcv_filename)
        self._outlarcv.initialize()
        self._log = logging.getLogger(__name__)

        FixedCROIFromFlash = ublarcvapp.UBSplitDetector
        self._infill_tree_name  = infill_tree_name
        self._adc_producer      = adc_producer
        self._chstatus_producer = chstatus_producer

        self._ubsplitdet = None

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
        ev_wholeview = self._inlarcv.get_data(larcv.kProductImage2D,
                                              self._adc_producer)
        ev_chstatus = self._inlarcv.get_data(larcv.kProductChStatus,
                                                self._chstatus_producer)
        wholeview_v = ev_wholeview.Image2DArray()
        print("Wholeview meta: ",wholeview_v[0].meta().dump())
        labels_v = ev_wholeview.Image2DArray()
        nplanes = wholeview_v.size()
        run    = self._inlarcv.event_id().run()
        subrun = self._inlarcv.event_id().subrun()
        event  = self._inlarcv.event_id().event()
        print("num of planes in entry {}: ".format((run,subrun,event)),nplanes)

        # crop using UBSplit for infill network
        # we want to break the image into set crops to send in

        # define the bbox_v images and cropped images
        bbox_list = larcv.EventROI()
        img2d_list = larcv.EventImage2D()

        bbox_v = larcv.EventROI().ROIArray()
        img2d_v = larcv.EventImage2D().Image2DArray()

        # we split the entire image using UBSplitDetector
        scfg="""Verbosity: 3
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
        MaxRandomAttempts: 4
        MinFracPixelsInCrop: -0.0001
        TickForward: true
        """

        fcfg = open("ubsplit.cfg",'w')
        print(scfg,end="",file=fcfg)
        fcfg.close()


        cfg = larcv.CreatePSetFromFile( "ubsplit.cfg", "UBSplitDetector" )
        algo = ublarcvapp.UBSplitDetector()
        algo.initialize()
        algo.configure(cfg)
        algo.set_verbosity(2)

        bbox_list = larcv.EventROI()
        img2du = []
        img2dv = []
        img2dy = []
        img2d_list = []

        bbox_v = larcv.EventROI().ROIArray()
        img2d_v = larcv.EventImage2D().Image2DArray()

        algo.process( wholeview_v,img2d_v,bbox_v )

        print("crop meta: ",img2d_v[0].meta().dump())

        # seperate by planes
        for i in img2d_v:
            p = i.meta().plane()
            if p == 0:
                img2du.append(i)
            elif p == 1:
                img2dv.append(i)
            elif p == 2:
                img2dy.append(i)

        img2d_list.append(img2du)
        img2d_list.append(img2dv)
        img2d_list.append(img2dy)

        for plane in img2d_list:
            print("In list" , len(plane))


        # send messages
        # (send crops to worker to go through network)
        replies = self.send_image_list(img2d_list,run=run,subrun=subrun,event=event)
        print ("FINISHED SEND STEP")
        self.process_received_images(wholeview_v,ev_chstatus,replies)
        print ("FINISHED PROCESS STEP")

        self._outlarcv.set_id( self._inlarcv.event_id().run(),
                               self._inlarcv.event_id().subrun(),
                               self._inlarcv.event_id().event())

        self._outlarcv.save_entry()
        print("SAVED ENTRY")
        return True


    def send_image_list(self,img2d_list, run=0, subrun=0, event=0):
        """ send all images in an event to the worker and receive msgs"""
        planes = [0,1,2]

        planes.sort()
        rse = (run,subrun,event)
        self._log.info("sending images with rse={}".format(rse))

        imgout_v = {}
        nsize_uncompressed = 0
        nsize_compressed = 0
        received_compressed = 0
        received_uncompressed = 0
        imageid_received = {}
        nimages_sent = 0
        for p in planes:
            print("plane ", p)
            if p not in imgout_v:
                imgout_v[p] = []


            self._log.info("sending images in plane[{}]: num={}."
                            .format(p,len(img2d_list[p])))

            # send image
            msg = []
            for img2d in img2d_list[p]:
                img_id = nimages_sent # make an id to track if it comes back
                bson = larcv.json.as_pystring(img2d,
                                              run, subrun, event, img_id)
                nsize_uncompressed += len(bson)
                compressed = zlib.compress(bson)
                nsize_compressed   += len(compressed)
                msg.append(compressed)
                # we make a flag to mark if we got this back
                imageid_received[img_id] = False
                nimages_sent += 1
            self.send("infill_plane%d"%(p),*msg)
            print("sent plane to worker!")

            # receives
            isfinal = False
            while not isfinal:
                workerout = self.recv_part()
                isfinal =  workerout is None
                if isfinal:
                    self._log.info("received done indicator by worker")
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
                    replyimg = larcv.json.image2d_from_pystring(data,
                                c_run, c_subrun, c_event, c_id )
                        #byref(c_run),byref(c_subrun),byref(c_event),byref(c_id))
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

                    imgout_v[p].append(replyimg)
                self._log.debug("running total, converted plane[{}] images: {}"
                                .format(p,len(imgout_v[p])))
            complete = True
            # should be a multiple of 2: (shower,track)
            if len(imgout_v[p])>0 and len(imgout_v[p])%2!=0:
                complete = False
            # should have got all images back
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
        return imgout_v

    def process_received_images(self, wholeview_v,ev_chstatus, outimg_v):
        """ receive the list of images from the worker """
        # this is where we stitch the crops together
        nplanes = wholeview_v.size()

        ev_infill = self._outlarcv.\
                        get_data(larcv.kProductImage2D,
                                 self._infill_tree_name)
        ev_input = self._outlarcv.\
                        get_data(larcv.kProductImage2D,
                                self._adc_producer)

        for p in xrange(nplanes):

            # create final output image
            outputimg = larcv.Image2D( wholeview_v.at(p).meta() )
            outputimg.paint(0)

            # temp image to use for averaging later
            overlapcountimg = larcv.Image2D( wholeview_v.at(p).meta() )
            overlapcountimg.paint(0)

            nimgsets = len(outimg_v[p])
            output_meta=outputimg.meta()

            # loop through all crops to stitch onto outputimage
            for iimgset in xrange(nimgsets):
                ublarcvapp.InfillImageStitcher().Croploop(output_meta,
                                            outimg_v[p][iimgset], outputimg,
                                            overlapcountimg)
            # creates overlay image and takes average where crops overlapped
            ublarcvapp.InfillImageStitcher().Overlayloop(p,output_meta,outputimg,
                                            overlapcountimg, wholeview_v, ev_chstatus)

            ev_infill.Append(outputimg)
            ev_input.Append(wholeview_v.at(p))

    def process_entries(self,start=0, end=-1):
        if end<0:
            end = self.get_entries()-1

        for ientry in xrange(start,end+1):
            self.process_entry(ientry)

    def finalize(self):
        self._inlarcv.finalize()
        self._outlarcv.finalize()
