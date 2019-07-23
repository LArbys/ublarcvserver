from __future__ import print_function
import numpy

import logging
from ublarcvserver import Client
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
import zlib
from ctypes import c_int, byref
from ROOT import std


#larcv.load_pyutils()
larcv.json.load_jsonutils()

class UBInfillSparseClient(Client):

    def __init__(self, broker_address,
                 larcv_supera_file,
                 output_larcv_filename,
                 adc_producer="wire", chstatus_producer="wire",
                 tick_backwards=False, infill_tree_name="infill",
                 save_adc_image=False,
                 use_compression=False,
                 **kwargs):
        """
        """
        super(UBInfillSparseClient,self).__init__(broker_address,**kwargs)

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
        self._use_compression = use_compression
        self._save_adc_image = save_adc_image

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
        # ev_wholeview_copy = larcv.EventImage2D()
        labels = larcv.EventImage2D()
        labels.Append(larcv.Image2D(wholeview_v[0].meta()))
        labels.Append(larcv.Image2D(wholeview_v[1].meta()))
        labels.Append(larcv.Image2D(wholeview_v[2].meta()))
        labels_v = labels.Image2DArray()
        # labels_v = [larcv.Image2D(wholeview_v[0].meta()),larcv.Image2D(wholeview_v[1].meta()),larcv.Image2D(wholeview_v[2].meta())]
        nplanes = wholeview_v.size()
        run    = self._inlarcv.event_id().run()
        subrun = self._inlarcv.event_id().subrun()
        event  = self._inlarcv.event_id().event()
        print("num of planes in entry {}: ".format((run,subrun,event)),nplanes)

        # crop using UBSplit for infill network
        # we want to break the image into set crops to send in

        # create labels_image_v
        # for img in labels_v:
        #     img.paint(0)

        labels_v = ublarcvapp.InfillDataCropper().ChStatusToLabels(labels_v,ev_chstatus)

        # we split the entire image using UBSplitDetector
        scfg="""Verbosity: 3
        InputProducer: \"wire\"
        OutputBBox2DProducer: \"detsplit\"
        CropInModule: true
        OutputCroppedProducer: \"detsplit\"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 496
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: -1
        RandomizeCrops: 0
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

        bbox_labels_list = larcv.EventROI()
        img2du_labels = []
        img2dv_labels = []
        img2dy_labels = []
        img2d_labels_list = []
        bbox_labels_v = larcv.EventROI().ROIArray()
        img2d_labels_v = larcv.EventImage2D().Image2DArray()
        algo.process( labels_v,img2d_labels_v,bbox_labels_v )

        algo.finalize()

        # seperate by planes
        for i in img2d_v:
            p = i.meta().plane()
            if p == 0:
                img2du.append(i)
            elif p == 1:
                img2dv.append(i)
            elif p == 2:
                img2dy.append(i)

        for i in img2d_labels_v:
            p = i.meta().plane()
            if p == 0:
                img2du_labels.append(i)
            elif p == 1:
                img2dv_labels.append(i)
            elif p == 2:
                img2dy_labels.append(i)

        img2d_list.append(img2du)
        img2d_list.append(img2dv)
        img2d_list.append(img2dy)
        img2d_labels_list.append(img2du_labels)
        img2d_labels_list.append(img2dv_labels)
        img2d_labels_list.append(img2dy_labels)

        for plane in img2d_list:
            print("In list" , len(plane))
        for plane in img2d_labels_list:
            print("In labels list" , len(plane))

        # sparsify image 2d
        thresholds = std.vector("float")(1,10.0)
        sparseimg_list = []
        usparse_v = []
        vsparse_v = []
        ysparse_v = []

        for a,b in zip(img2d_list, img2d_labels_list):
            for img, label in zip(a,b):
                p = img.meta().plane()
                sparse_img = larcv.SparseImage(img,label,thresholds)
                if (p ==0):
                    usparse_v.append(sparse_img)
                elif (p ==1):
                    vsparse_v.append(sparse_img)
                elif (p ==2):
                    ysparse_v.append(sparse_img)

        sparseimg_list.append(usparse_v)
        sparseimg_list.append(vsparse_v)
        sparseimg_list.append(ysparse_v)

        for plane in sparseimg_list:
            print("In sparse list" , len(plane))


        # send messages
        # (send crops to worker to go through network)
        replies = self.send_image_list(sparseimg_list,run=run,subrun=subrun,event=event)
        print ("FINISHED SEND STEP")
        self.process_received_images(wholeview_v,ev_chstatus,replies,img2d_list)
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
                img_id = nimages_sent
                # print(len(img2d.meta_v()), " ", run, " ",subrun," ",event, " ",img_id)
                # make an id to track if it comes back
                bson = larcv.json.as_bson_pybytes(img2d,run, subrun, event, img_id)
                # print ("made bson")
                nsize_uncompressed += len(bson)
                if self._use_compression:
                    compressed = zlib.compress(bson)
                    self._log.info("Compressing bson: {} to {} bytes".format(len(bson),len(compressed)))
                else:
                    compressed = bson
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
                workerout = self.recv_part(timeout=300) # 5 minute wait
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
                    if self._use_compression:
                        data = zlib.decompress(data)
                    received_uncompressed += len(data)
                    c_run = c_int()
                    c_subrun = c_int()
                    c_event = c_int()
                    c_id = c_int()
                    replyimg = larcv.json.sparseimg_from_bson_pybytes(data,
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

    def process_received_images(self, wholeview_v,ev_chstatus, outimg_v,img2d_list):
        """ receive the list of images from the worker """
        # first change sparse to dense
        outdense_v =[]
        denseu = []
        densev = []
        densey = []
        for plane in xrange(len(outimg_v)):
            for img in xrange(len(outimg_v[plane])):
                predimg = larcv.Image2D(img2d_list[plane][img].meta())
                predimg = outimg_v[plane][img].as_Image2D()
                if plane ==0:
                    denseu.append(predimg)
                elif plane ==1:
                    densev.append(predimg)
                elif plane ==2:
                    densey.append(predimg)
        outdense_v.append(denseu)
        outdense_v.append(densev)
        outdense_v.append(densey)
        print("size of dense array",len(outdense_v))

        # this is where we stitch the crops together
        nplanes = wholeview_v.size()

        ev_infill = self._outlarcv.\
                        get_data(larcv.kProductImage2D,
                                 self._infill_tree_name)

        if self._save_adc_image:
            # save a copy of input image
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
            # print ("nimgsets",nimgsets)
            output_meta=outputimg.meta()

            # loop through all crops to stitch onto outputimage
            for iimg in xrange(len(outdense_v[p])):
                # print (len(outdense_v[p][iimg]))
                ublarcvapp.InfillImageStitcher().Croploop(output_meta,
                                            outdense_v[p][iimg][0], outputimg,
                                            overlapcountimg)
            # creates overlay image and takes average where crops overlapped
            ublarcvapp.InfillImageStitcher().Overlayloop(p,output_meta,outputimg,
                                            overlapcountimg, wholeview_v, ev_chstatus)

            ev_infill.Append(outputimg)
            if self._save_adc_image:
                ev_input.Append(wholeview_v.at(p))

    def process_entries(self,start=0, end=-1):
        if end<0:
            end = self.get_entries()-1

        for ientry in xrange(start,end+1):
            self.process_entry(ientry)

    def finalize(self):
        self._inlarcv.finalize()
        self._outlarcv.finalize()
