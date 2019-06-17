import logging
from ublarcvserver import Client
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from ROOT import std
import zlib
from ctypes import c_int, byref

larcv.json.load_jsonutils()

class UBSparseLArFlowClient(Client):

    def __init__(self, broker_address,
                 larcv_supera_file,
                 output_larcv_filename,
                 adc_producer="wire",
                 sparseimage_input_producer="larflow",
                 sparseimage_output_producer="dualflow",
                 has_sparseimage_data=False,
                 save_as_sparseimg=False,
                 tick_backwards=False,
                 use_compression=False,
                 cropper_cfg="ubcrop.cfg",
                 **kwargs):
        """
        this class loads either larcv::sparseimage or larcv::image2d data from
        the input file, prepares the data into a binary json (bson) message to
        be sent to the broker. when the broker replies with worker output,
        save it to larcv root file.
        """
        super(UBSparseLArFlowClient,self).__init__(broker_address,**kwargs)

        # setup the input iomanager
        tick_direction = larcv.IOManager.kTickForward
        if tick_backwards:
            tick_direction = larcv.IOManager.kTickBackward

        # setup splitter: for processing wholeview images
        if not has_sparseimage_data:
            self.splitter = larcv.ProcessDriver( "ProcessDriver" )
            self.splitter.configure( cropper_cfg )
            infiles = std.vector("std::string")()
            infiles.push_back( larcv_supera_file )
            self.splitter.override_input_file(infiles)
            self.splitter.initialize()
            self._inlarcv = self.splitter.io_mutable()            
        else:
            self._inlarcv = larcv.IOManager(larcv.IOManager.kREAD,"",
                                            tick_direction)
            self._inlarcv.add_in_file(larcv_supera_file)
            self._inlarcv.initialize()

        # setup output iomanager
        self._outlarcv = larcv.IOManager(larcv.IOManager.kWRITE)
        self._outlarcv.set_out_file(output_larcv_filename)
        self._outlarcv.initialize()

        # setup config
        self._adc_producer = adc_producer
        self._sparseimage_input_producer=sparseimage_input_producer
        self._sparseimage_output_producer=sparseimage_output_producer
        self._use_sparseimage_data=has_sparseimage_data
        self._save_as_sparseimg=save_as_sparseimg
        self._use_compression = use_compression

        # thresholds: adc values must be above this value to be included
        self._threshold_v   = std.vector("float")(3,10.0)
        # cuton flag: if pixel passes in the given plane, we save values in all three
        #             this is required because of the submanifold structure
        self._cuton_pixel_v = std.vector("int")(3,1)


        # setup logger
        self._log = logging.getLogger(__name__)

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
        # we either split the ADC image or get cropped sparseimage data
        if self._use_sparseimage_data:
            # load sparseimage data
            ev_sparseimg = self._inlarcv.get_data(larcv.kProductSparseImage,
                                                  self._sparseimage_input_producer)
            sparseimg_v = ev_sparseimg.SparseImageArray()
        else:
            # we must split the image
            self.splitter.process_entry( entry_num, False, False )
            ev_cropped = self.splitter.io_mutable().get_data(larcv.kProductImage2D, "detsplit")
            cropped_v = ev_cropped.Image2DArray()
            self._log.debug("cropped wholeview image into {} crop sets (total: {})".format(cropped_v.size()/3,cropped_v.size()))
            sparseimg_v = std.vector("larcv::SparseImage")()
            for iset in xrange( cropped_v.size()/3 ):
                sparsedata = larcv.SparseImage( cropped_v, iset*3, iset*3+2, self._threshold_v, self._cuton_pixel_v )
                sparseimg_v.push_back( sparsedata )
            
        self._log.debug("prepared {} sparse numpy arrays for processing.".format( sparseimg_v.size() ))

        # get whole adc image
        #ev_wholeview = self._inlarcv.get_data(larcv.kProductImage2D,
        #                                      self._adc_producer)
        #wholeview_v = ev_wholeview.Image2DArray()
        #nplanes = wholeview_v.size()
        run    = self._inlarcv.event_id().run()
        subrun = self._inlarcv.event_id().subrun()
        event  = self._inlarcv.event_id().event()
        print "prcessing entry {}: ".format((run,subrun,event))


        # send messages
        if sparseimg_v.size()>0:
            replies = self.send_sparseimages(sparseimg_v,run=run,subrun=subrun,event=event)
            if len(replies)>0:
                self.process_received_images(replies)
        else:
            # create blank
            evimg_outblank = self._outlarcv.get_data(larcv.kProductImage2D,
                                                     self._sparseimage_output_producer)
            if self._save_as_sparseimg:
                evsp_blank = self._outlarcv.get_data(larcv.kProductSparseImage,
                                                     self._sparseimage_output_producer)


        self._outlarcv.set_id( self._inlarcv.event_id().run(),
                               self._inlarcv.event_id().subrun(),
                               self._inlarcv.event_id().event())

        self._outlarcv.save_entry()
        return True

    def send_sparseimages(self,sparseimg_v, run=0, subrun=0, event=0):
        """
        process all images in sparseimg_v vector

        return list of sparseimage vectors containing replies for each input sparseimage
        """
        imgout_vv = []
        for iimg in xrange(sparseimg_v.size()):
            imgout_v = self.send_sparsedata( sparseimg_v.at(iimg), run, subrun, event )
            imgout_vv.append( imgout_v )
        return imgout_vv

    def send_sparsedata(self,sparsedata, run=0, subrun=0, event=0):
        """
        send all images in an event to the worker and receive msgs
        run, subrun, event ID used to make sure returning message is correct one

        inputs
        ------
        sparsedata dictionary with data to be sent. 
                   primarily sparsedata["pixdata"] contaning list of SparseImage objects.
        run int run number to store in message
        subrun int subrun number to store in message
        event int event number to store in messsage
        """

        msg = [] # message components
        bson = larcv.json.as_bson_pybytes(sparsedata,
                                           run, subrun, event, 0)
        nsize_uncompressed = len(bson)
        if self._use_compression:
            compressed = zlib.compress(bson)
        else:
            compressed = bson
        nsize_compressed   = len(compressed)
        msg.append(compressed)
        rse = (run,subrun,event)

        # we make a flag to mark if we got this back
        imageid_received = False
        self.send("ublarflow_plane%d"%(2),*msg)

        # receives
        isfinal = False
        received_compressed = 0
        received_uncompressed = 0
        imgout_v = []
        while not isfinal:
            workerout = self.recv_part()
            isfinal =  workerout is None
            if isfinal:
                self._log.debug("received done indicator by worker")
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
                rep_rse = (c_run.value,c_subrun.value,c_event.value)
                self._log.debug("rec images with rse={}".format(rep_rse))
                if rep_rse!=rse:
                    self._log.warning("image with wronge rse={}. ignoring."
                                .format(rep_rse))
                    continue

                imgout_v.append(replyimg)
                self._log.debug("running total, converted images: {}"
                                .format(len(imgout_v)))
            complete = True

            if len(imgout_v)!=1:
                complete = False

            # should have got all images back
            if not complete:
                raise RuntimeError(\
                    "Did not receive complete set for all images")

        self._log.debug("Total sent size. uncompressed=%.2f MB compreseed=%.2f"\
                        %(nsize_uncompressed/1.0e6,nsize_compressed/1.0e6))
        self._log.debug("Total received. compressed=%.2f uncompressed=%.2f MB"\
                        %(received_compressed/1.0e6, received_uncompressed/1.0e6))
        return imgout_v

    def process_received_images(self, sparseimg_vv):
        """ receive the list of images from the worker """

        evimg_output = self._outlarcv.\
                        get_data(larcv.kProductImage2D,
                                 self._sparseimage_output_producer)

        if self._save_as_sparseimg:
            evsparse_output = self._outlarcv.\
                        get_data(larcv.kProductSparseImage,
                                self._sparseimage_output_producer)
        
        # convert into images
        for sparseimg_v in sparseimg_vv:
            for sparseimg in sparseimg_v:
                dense_img_v = sparseimg.as_Image2D()
                for iimg in xrange(dense_img_v.size()):
                    evimg_output.Append( dense_img_v.at(iimg) )
                if self._save_as_sparseimg:
                    evsparse_output.Append( sparseimg )

        return True

    def process_entries(self,start=0, end=-1):
        if end<0:
            end = self.get_entries()-1

        for ientry in xrange(start,end+1):
            self.process_entry(ientry)

    def finalize(self):
        self._inlarcv.finalize()
        self._outlarcv.finalize()
