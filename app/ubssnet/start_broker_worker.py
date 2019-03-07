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


def startup_broker_and_workers( broker_address, broker_bindpoint,
                                weights_files, batch_size=1,
                                nplanes=[0,1,2]):

    logging.basicConfig(level=logging.DEBUG)
    pbroker = Process(target=start_ubssnet_broker,
                      args=(broker_bindpoint,))
    pbroker.daemon = True
    pbroker.start()

    # setup the worker
    pworkers = []
    for p in nplanes:
        pworker = Process(target=start_ubssnet_worker,
                          args=(broker_address,p,weights_files[p],"cuda",batch_size))
        pworker.daemon = True
        pworkers.append(pworker)

    for pworker in pworkers:
        pworker.start()

    return pbroker, pworkers

if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "/home/twongjirad/working/nutufts/pytorch-uresnet/weights/"
    weights_files = {2:weights_dir+"/saved_caffe_weights_plane2.tar"}

    logging.basicConfig(level=logging.DEBUG)

    pbroker, pworkers = startup_broker_and_workers(endpoint,bindpoint,
                                                    weights_files,nplanes=[2])

    print "[ENTER] to quit."
    raw_input()
