import logging
from ublarcvserver import Client
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp

#larcv.load_pyutils()
larcv.json.load_jsonutils()

class UBSSNetClient(Client):

    def __init__(self, broker_address,
                    larcv_supera_file, image2d_tree_name,
                    output_larcv_filename,
                    larlite_opreco_file=None, apply_opflash_roi=False,
                    tick_backwards=False, ssnet_tree_name="ssnet",
                    intimewin_min_tick=190, intimewin_max_tick=320,**kwargs):
        """
        """
        super(UBSSNetClient,self).__init__(broker_address,**kwargs)
        tick_direction = larcv.IOManager.kTickForward
        if tick_backwards:
            tick_direction = larcv.IOManager.kTickBackward
        self._inlarcv = larcv.IOManager(larcv.IOManager.kREAD,"",
                                        tick_direction)
        self._inlarcv.add_in_file(larcv_supera_file)
        self._inlarcv.initialize()

        LArliteManager = ublarcvapp.LArliteManager
        self._inlarlite = None
        if larlite_opreco_file is not None:
            self._inlarlite = LArliteManager(larlite.storage_manager.kREAD)
            self._inlarlite.add_in_filename(larlite_opreco_file)
            self._inlarlite.open()

        self._outlarcv = larcv.IOManager(larcv.IOManager.kWRITE)
        self._outlarcv.set_out_file(output_larcv_filename)
        self._outlarcv.initialize()
        self._log = logging.getLogger(__name__)


        FixedCROIFromFlash = ublarcvapp.ubdllee.FixedCROIFromFlashAlgo
        self._ssnet_tree_name = ssnet_tree_name
        self._apply_opflash_roi = apply_opflash_roi
        if self._apply_opflash_roi:
            self._croi_fromflash_algo = FixedCROIFromFlash()
            self._intimewin_min_tick = intimewin_min_tick
            self._intimewin_max_tick = intimewin_max_tick
        else:
            self._ubsplitdet = None

    def process_entry(self,entry_num):
        """ perform all actions -- send, receive, process, store -- for entry"""

        # get the entries
        ok = self._inlarcv.read_entry(entry_num)
        if not ok:
            raise RuntimeError("could not read larcv entry %d"%(entry_num))

        # define the roi_v images
        roi_v = []
        if self._inlarlite and self._apply_opflash_roi:
            # use the intime flash to look for a CROI
            self._inlarlite.syncEntry(self._inlarcv)
            ev_opflash = self._inlarlite.get_data(larlite.data.kOpFlash,
                                                    "simpleFlashBeam")
            nintime_flash = 0
            for iopflash in xrange(ev_opflash.size()):
                opflash = ev_opflash.at(iopflash)
                if ( opflash.Time()<self._intimewin_min_tick*0.015625
                    or opflash.Time()>self._intimewin_max_tick*0.015625):
                    continue
                flashrois=self._croi_fromflash_algo.findCROIfromFlash(opflash);
                for iroi in xrange(flashrois.size()):
                    roi_v.append( flashrois.at(iroi) )
                nintime_flash += 1
            print "number of intime flashes: ",nintime_flash
        else:
            # we split the entire image
            raise RuntimeError("Use of ubsplitdet for image not implemented")

        ev_wholeview = self._inlarcv.get_data(larcv.kProductImage2D,"wire")
        wholeview_v = ev_wholeview.Image2DArray()
        nplanes = wholeview_v.size()

        print "number of planes: ",nplanes

        # make crops from the roi_v
        img2d_v = {}
        for plane in xrange(nplanes):
            if plane not in img2d_v:
                img2d_v[plane] = []

        for roi in roi_v:
            for plane in xrange(nplanes):
                if plane not in img2d_v:
                    img2d_v[plane] = []

                wholeimg = wholeview_v.at(plane)
                bbox = roi.BB(plane)
                img2d = wholeimg.crop( bbox )

                img2d_v[plane].append( img2d )

        planes = img2d_v.keys()
        planes.sort()
        print "Number of images on each plane:"
        for plane in planes:
            print "plane[{}]: {}".format(plane,len(img2d_v[plane]))

        # send messages
        replies = self.send_image_list(img2d_v)
        self.process_received_images(wholeview_v,replies)

        self._outlarcv.set_id( self._inlarcv.event_id().run(),
                               self._inlarcv.event_id().subrun(),
                               self._inlarcv.event_id().event())

        self._outlarcv.save_entry()


    def send_image_list(self,img2d_list):
        """ send all images in an event to the worker and receive msgs"""
        planes = img2d_list.keys()
        planes.sort()
        self._log.debug("sending images from {} planes.".format(planes))
        imgout_v = {}
        for p in planes:
            if p not in imgout_v:
                imgout_v[p] = []

            if p not in [2]:
                continue


            self._log.debug("images in plane[{}]: {}."
                            .format(p,len(img2d_list[p])))

            # send image
            msg = []
            for img2d in img2d_list[p]:
                print img2d
                bson = larcv.json.as_pystring(img2d)
                msg.append(bson)
            self.send("ubssnet_plane%d"%(p),*msg)

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
                for reply in workerout:
                    replyimg = larcv.json.image2d_from_pystring(str(reply))
                    imgout_v[p].append(replyimg)
                    self._log.debug("total converted plane[{}] images: {}"
                                    .format(p,len(imgout_v[p])))

            # should be a multiple of 3 (track,shower,bg)
            if len(imgout_v[p])>0 and len(imgout_v[p])%3!=0:
                raise RuntimeError(\
                    "Did not receive complete set for all images")

        return imgout_v

    def process_received_images(self, wholeview_v, outimg_v):
        """ receive the list of images from the worker """
        nplanes = wholeview_v.size()

        ev_shower = self._outlarcv.\
                        get_data(larcv.kProductImage2D,"ssnet_shower")
        ev_track  = self._outlarcv.\
                        get_data(larcv.kProductImage2D,"ssnet_track")
        ev_bg     = self._outlarcv.\
                        get_data(larcv.kProductImage2D,"ssnet_background")

        for p in xrange(nplanes):

            showerimg = larcv.Image2D( wholeview_v.at(p).meta() )
            bgimg     = larcv.Image2D( wholeview_v.at(p).meta() )
            trackimg  = larcv.Image2D( wholeview_v.at(p).meta() )
            showerimg.paint(0)
            bgimg.paint(0)
            trackimg.paint(0)

            nimgsets = len(outimg_v[p])/3
            for iimgset in xrange(nimgsets):
                bg  = outimg_v[p][iimgset*3+0]
                shr = outimg_v[p][iimgset*3+1]
                trk = outimg_v[p][iimgset*3+2]

                showerimg.overlay(shr,larcv.Image2D.kOverWrite)
                trackimg.overlay( trk,larcv.Image2D.kOverWrite)
                bgimg.overlay(    bg, larcv.Image2D.kOverWrite)
            ev_shower.Append(showerimg)
            ev_track.Append(trackimg)
            ev_bg.Append(bgimg)

    def finalize(self):
        self._inlarcv.finalize()
        self._outlarcv.finalize()
        self._inlarlite.close()
