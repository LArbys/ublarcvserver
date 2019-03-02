import os,sys,time
from ublarcvserver import ublarcvserver
from multiprocessing import Process
from larcv import larcv
from ROOT import std

"""
This script is used to test the Majordomo classes.

We implement a dummy setup where the client and worker just say hello to each other.
Also servers as an example.

We also setup the basic larcv client and worker, which pass larcv images back and forth.
"""
verbose = False

def start_worker( endpoint ):
    global verbose
    print "start worker on ",endpoint
    worker = ublarcvserver.MirrorWorker(endpoint, verbose)
    print "worker started: ",worker.get_id_name()
    worker.run()
    #while 1:
    #    time.sleep(1)


# endpoint:
endpoint  = "tcp://localhost:6005"
bindpoint = "tcp://*:6005"

# SETUP THE LARCV INPUT
input_rootfile = sys.argv[1]
io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file( input_rootfile )
io.initialize()

# setup the worker
pworker = Process(target=start_worker,args=(endpoint,))
pworker.daemon = True
pworker.start()
#worker = ublarcvserver.DummyWorker(endpoint, True)
print "worker process created"

# setup the broker
broker = ublarcvserver.MDBroker(bindpoint, verbose)
broker.start()
print "broker started"

# setup the client
client = ublarcvserver.LArCVClient(endpoint, "mirror", verbose, ublarcvserver.LArCVClient.kSPARSE)
print "client created"

for x in xrange(5):
    io.read_entry(x)
    print "REQUEST %d"%(x+1)

    # get images from tree
    event_images = io.get_data(larcv.kProductImage2D,"wire")

    # load images into client
    for iimg in xrange(event_images.Image2DArray().size()):
      client.addImageAsPixelList( event_images.Image2DArray().at(iimg), 10.0 )

    # send images and get reply
    client.request()
    time.sleep(1)

    # get the images back
    reply_img_v = std.vector("larcv::Image2D")()
    client.takeImages( reply_img_v )
    print "returned %d images"%(reply_img_v.size())
    for iimg in xrange(reply_img_v.size()):
        print " img[{}] {}".format(iimg,reply_img_v.at(iimg).meta().dump())


print "[ENTER] to end"
raw_input()
