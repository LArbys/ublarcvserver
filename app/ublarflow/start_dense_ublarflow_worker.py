import os,sys,logging,time
from multiprocessing import Process
from UBDenseLArFlowWorker import UBDenseLArFlowWorker
from ublarcvserver import start_broker

"""
Start the broker and worker for Dense LArFlow Network.
Provides utility function for starting worker(s).
"""

def start_dense_ublarflow_worker(broker_address,flow_dir,weight_file,
                                 device,batch_size,use_half,
                                 ssh_thru_server,ssh_password):
    """
    start single copy of dense larflow worker
    """
    worker=UBDenseLArFlowWorker(broker_address,flow_dir,weight_file,
                                device,batch_size,
                                use_half=use_half,
                                ssh_thru_server=ssh_thru_server,
                                ssh_password=ssh_password)
    worker.connect()
    print "worker started: ",worker.idname()
    worker.run()

def start_daemon_ublarflow_worker(broker_address,flow_dir,weight_file,
                                  device,batch_size,use_half,
                                  ssh_thru_server,ssh_password):
    pworker = Process(target=start_dense_ublarflow_worker,
                      args=(broker_address,flow_dir,weight_file,
                            device,batch_size,use_half,
                            ssh_thru_server,ssh_password))
    pworker.daemon = True
    pworker.start()
    return pworker



if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6666"
    bindpoint = "tcp://*:6666"
    weights_dir = "../../networks/larflow/weights/"
    weights_files = {"y2u":weights_dir+"/devfiltered_larflow_y2u_832x512_32inplanes.tar",
                     "y2v":weights_dir+"/devfiltered_larflow_y2v_832x512_32inplanes.tar"}
    
    if len(sys.argv)==2:
        device = sys.argv[1]
    else:
        device = "cuda"


    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworker = start_daemon_ublarflow_worker(endpoint,'y2u',
                                            weights_files['y2u'],
                                            device,1,False,None,None)
    pworker = start_daemon_ublarflow_worker(endpoint,'y2v',
                                            weights_files['y2v'],
                                            device,1,False,None,None)


    print "[CTRL+C] to quit."
    while True:
        time.sleep(1)
        
