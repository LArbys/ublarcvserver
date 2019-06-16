import os,sys,logging
from multiprocessing import Process
from UBSSNetWorker import UBSSNetWorker
from ublarcvserver import start_broker

"""
Start the broker and worker. Run one client. Useful for tests.
"""

def start_ubssnet_worker(broker_address,plane,weight_file,
                         device,batch_size,
                         ssh_thru_server,ssh_password):
    print batch_size,type(batch_size)
    worker=UBSSNetWorker(broker_address,plane,weight_file,
                         device,batch_size,
                         ssh_thru_server=ssh_thru_server,
                         ssh_password=ssh_password)
    worker.connect()
    print "worker started: ",worker.idname()
    worker.run()


def startup_ubssnet_workers( broker_address, weights_files,
                             devices=["cuda","cuda","cuda"],
                             batch_size=1,
                             nplanes=[0,1,2],
                             ssh_thru_server=None, ssh_password=None,
                             start=True):
    if type(devices) is str:
        devices = len(nplanes)*[devices]
    if len(devices)>len(nplanes):
        devices = [devices[x] for x in xrange(len(nplanes))]
    elif len(devices)<len(nplanes):
        raise ValueError("devices need to be speficied for each plane")

    # setup the worker
    pworkers = []
    print "plans: ",nplanes
    print "devices: ",devices
    for p,device in zip(nplanes,devices):
        pworker = Process(target=start_ubssnet_worker,
                          args=(broker_address,p,weights_files[p],
                                device,batch_size,
                                ssh_thru_server,ssh_password))
        pworker.daemon = True
        pworkers.append(pworker)

    if start:
        for pworker in pworkers:
            pworker.start()

    return pworkers




if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6000"
    bindpoint = "tcp://*:6000"
    #weights_dir = "/home/twongjirad/working/nutufts/pytorch-uresnet/weights/"
    weights_dir = "/home/taritree/working/larbys/ubdl/ublarcvserver/networks/pytorch-uresnet/weights/"
    weights_files = {0:weights_dir+"/mcc8_caffe_ubssnet_plane0.tar",
                     1:weights_dir+"/mcc8_caffe_ubssnet_plane1.tar",
                     2:weights_dir+"/mcc8_caffe_ubssnet_plane2.tar"}

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworkers = startup_ubssnet_workers(endpoint,weights_files,nplanes=[0,1,2])

    print "[ENTER] to quit."
    raw_input()
