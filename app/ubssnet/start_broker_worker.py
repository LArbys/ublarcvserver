import os,sys,logging
from multiprocessing import Process

from UBSSNetWorker import UBSSNetWorker
from ublarcvserver import Broker

"""
Start the broker and worker. Run one client. Useful for tests.
"""

def start_ubssnet_worker(broker_address,plane,weight_file,device,batch_size):
    worker=UBSSNetWorker(broker_address,plane,weight_file,device,batch_size)
    worker.connect()
    print "worker started: ",worker.idname()
    worker.run()

def start_ubssnet_broker(bindpoint):
    print "start broker"
    broker = Broker(bind=bindpoint)
    broker.run()
    print "broker closed"


if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "/home/twongjirad/working/nutufts/pytorch-uresnet/weights/"
    weights_file = weights_dir+"/saved_caffe_weights_plane2.tar"

    logging.basicConfig(level=logging.DEBUG)

    pbroker = Process(target=start_ubssnet_broker,args=(bindpoint,))
    pbroker.daemon = True
    pbroker.start()

    # setup the worker
    pworker = Process(target=start_ubssnet_worker,args=(endpoint,2,weights_file,"cuda",1))
    pworker.daemon = True
    pworker.start()
    print "worker process created"

    print "[ENTER] to quit."
    raw_input()
