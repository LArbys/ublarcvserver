import os,sys,logging
from multiprocessing import Process
from UBSparseLArFlowWorker import UBSparseLArFlowWorker
from ublarcvserver import start_broker

"""
Start the broker and worker. Run one client. Useful for tests.
"""

def start_ublarflow_worker(broker_address,plane,weight_file,
                           device,batch_size,
                           ssh_thru_server,ssh_password):
    """
    start single copy of larflow worker
    """
    worker=UBSparseLArFlowWorker(broker_address,plane,weight_file,
                         device,batch_size,
                         ssh_thru_server=ssh_thru_server,
                         ssh_password=ssh_password)
    worker.connect()
    print "worker started: ",worker.idname()
    worker.run()

def start_daemon_ublarflow_worker(broker_address,plane,weight_file,
                                  device,batch_size,
                                  ssh_thru_server,ssh_password):
    pworker = Process(target=start_ublarflow_worker,
                      args=(broker_address,plane,weight_file,
                            device,batch_size,
                            ssh_thru_server,ssh_password))
    pworker.daemon = True
    pworker.start()
    return pworker
    
    

if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6666"
    bindpoint = "tcp://*:6666"
    weights_dir = "../../networks/larflow/weights/"
    weights_files = {2:weights_dir+"/checkpoint.500th.tar"}

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworker = start_ublarflow_worker(endpoint,2,
                                     weights_files[2],"cpu",1,None,None)


    print "[ENTER] to quit."
    while True:
        time.sleep(1)
        raw_input()
