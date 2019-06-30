import os,sys,logging
from multiprocessing import Process
from UBInfillSparseWorker import UBInfillSparseWorker
from ublarcvserver import start_broker

"""
Start the broker and worker. Run one client. Useful for tests.
"""

def start_infill_worker(broker_address, plane,weight_file,
                         device, batch_size, use_compression,
                         ssh_thru_server, ssh_password):
    print(batch_size,type(batch_size))
    worker=UBInfillSparseWorker(broker_address,plane,weight_file,
                                device,batch_size,
                                use_compression=use_compression,
                                ssh_thru_server=ssh_thru_server,
                                ssh_password=ssh_password)
    worker.connect()
    print("worker started: ",worker.idname())
    worker.run()


def startup_infill_workers( broker_address,weights_file,
                             devices=["cuda","cuda","cuda"],
                             batch_size=1,
                             use_compression=False,
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
    print("planes: ",nplanes)
    print("devices: ",devices)
    for p,device in zip(nplanes,devices):
        pworker = Process(target=start_infill_worker,
                          args=(broker_address,p,weights_file[p],
                                device,batch_size, use_compression,
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
    weights_dir = "../../../networks/infill/"
    weights_files = {0:weights_dir+"/sparseinfill_uplane_test.tar",
                     1:weights_dir+"/sparseinfill_vplane_test.tar",
                     2:weights_dir+"/sparseinfill_yplane_test.tar"}

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworkers = startup_infill_workers(endpoint, weights, nplanes=[0,1,2])

    print("[ENTER] to quit.")
    raw_input()
