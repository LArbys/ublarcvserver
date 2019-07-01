import os,sys,logging
from multiprocessing import Process
from UBMRCNNWorker import UBMRCNNWorker
from ublarcvserver import start_broker

"""
Start the broker and worker. Run one client. Useful for tests.
"""

def start_ubmrcnn_worker(broker_address,plane,weight_file,
                         device_id,
                         batch_size,
                         ssh_thru_server,ssh_password):
    print(batch_size,type(batch_size))
    worker=UBMRCNNWorker(broker_address,plane,weight_file,
                         batch_size,
                         device_id=device_id,
                         ssh_thru_server=ssh_thru_server,
                         ssh_password=ssh_password,
                        )
    worker.connect()
    print("worker started: ",worker.idname())
    worker.run()


def startup_ubmrcnn_workers( broker_address, weights_files,
                             device_id=0,
                             batch_size=1,
                             nplanes=[0,1,2],
                             ssh_thru_server=None, ssh_password=None,
                             start=True):


    # setup the worker
    pworkers = []
    print("planes: ",nplanes)
    for p in nplanes:
        pworker = Process(target=start_ubmrcnn_worker,
                          args=(broker_address,p,weights_files[p],
                                device_id,batch_size,
                                ssh_thru_server,ssh_password))
        pworker.daemon = True
        pworkers.append(pworker)

    if start:
        for pworker in pworkers:
            pworker.start()

    return pworkers




if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "/home/jmills/workdir/ubdl/ublarcvserver/app/ubmrcnn"
    weights_files = {0:weights_dir+"/mcc8_mrcnn_plane0.pth",
                     1:weights_dir+"/mcc8_mrcnn_plane1.pth",
                     2:weights_dir+"/mcc8_mrcnn_plane2.pth"}

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworkers = startup_ubmrcnn_workers(endpoint,weights_files,nplanes=[0,1,2])


    print("[ENTER] to quit.")
    if sys.version_info[0] < 3:
        raw_input()
    else:
        input()
