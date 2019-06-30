#!/usr/bin/env python
import os,sys,logging,time,argparse
import getpass

"""
Provides script to start worker for Sparse LArFlow Network.

Also provides utility functions for starting worker(s).
"""

# hack prevent root from parsing args
tmp = list(sys.argv)
sys.argv = []
import ROOT
sys.argv = tmp

from multiprocessing import Process


def start_sparse_ublarflow_worker(broker_address,flow_dir,weight_file,
                                  device,batch_size,use_half,use_compression,
                                  ssh_thru_server,ssh_password):
    """
    start single copy of dense larflow worker

    inputs
    ------
    broker_address str Address of broker as "address:port". e.g. localhost:6000 or 111.222.333.444:6000
    flow_dir str Flow direction, either ['y2u','y2v']
    weight_file str Path to weight files
    device str String specifying device to launch on. e.g. "cuda:0" or "cpu"
    use_half bool If True, use half_precision when running network
    use_compression bool If True, expect messages to use zlib compression
    ssh_thru_server str Address of machine we should connect to via ssh
    ssh_password std Unprotected password to the machine
    """

    from UBSparseLArFlowWorker import UBSparseLArFlowWorker

    worker=UBSparseLArFlowWorker(broker_address,'Y',weight_file,
                                 device,batch_size,
                                 use_half=use_half, use_compression=use_compression,
                                 ssh_thru_server=ssh_thru_server,
                                 ssh_password=ssh_password)
    worker.connect()
    print "worker started: ",worker.idname()
    worker.run()

def start_daemon_ublarflow_worker(broker_address,flow_dir,weight_file,
                                  device,batch_size,use_half,use_compression,
                                  ssh_thru_server,ssh_password):

    """
    start single copy of dense larflow worker on a new process
    Note: arguments are the same as above function. Just passes through.

    return Process object in daemon mode running dense larflow worker object
    """
    
    pworker = Process(target=start_sparse_ublarflow_worker,
                      args=(broker_address,flow_dir,weight_file,
                            device,batch_size,use_half,use_compression,
                            ssh_thru_server,ssh_password))
    pworker.daemon = True
    pworker.start()
    return pworker



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("brokeraddr",type=str,help="Broker Address")
    parser.add_argument("-l","--logfile",type=str, default=None,
                        help="where the log file is writen to")
    parser.add_argument("-d","--debug",action="store_true",
                        help="set logger level to debug")
    parser.add_argument("-w","--weights-dir",type=str,default=None,
                        help="directory where weights can be found")
    parser.add_argument("-m","--device",type=str,default="cuda",
                        help="run with device. either 'cuda' or 'cpu'")
    parser.add_argument("-b","--batch-size",type=int,default=1,
                        help="batch size for each worker")
    parser.add_argument("-t","--ssh-tunnel",type=str,default=None,
                        help="Tunnel using SSH through the given IP address")
    parser.add_argument("-u","--ssh-user",type=str,default=None,
                        help="username for ssh tunnel command")
    parser.add_argument("-s","--launch-broker",action="store_true",default=False,
                        help="Launch a broker as well. Useful for testing.")
    parser.add_argument("-c","--use-compression",action="store_true",default=False,
                        help="Expect simple zlib compression on in-coming data.")
    
    args = parser.parse_args(sys.argv[1:])

    # Set the Logging Level
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    # Setup logger level
    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)
    else:
        logging.basicConfig(level=level)
        
    # get logger for this main function
    log = logging.getLogger("start_sparselarflow_worker_main")

    # set weight directionry
    weights_dir = "."
    if args.weights_dir is not None:
        weights_dir = args.weights_dir
    weights_files = {"dual":weights_dir+"/checkpoint.19400th.tar"}
    
    # set the device
    device = args.device

    # broker address
    broker_address = args.brokeraddr
    if "tcp://" not in broker_address:
        broker_address = "tcp://"+broker_address

    # if specified in arguments. start a broker
    if args.launch_broker:
        from ublarcvserver import start_broker
        bindpoint = broker_address.replace("localhost","*")
        log.info("START BROKER on: {}".format(bindpoint))
        pbroker = start_broker(bindpoint)
    else:
        pbroker = None

    # get the ssh url and password if specified in arguments
    if args.ssh_tunnel is not None:
        if args.ssh_user is None:
            raise ValueError("If using ssh tunnel, must provide user")
        print "Using ssh, please provide password"
        ssh_password =  getpass.getpass()
        ssh_url = "%s@%s"%(args.ssh_user,args.ssh_tunnel)
    else:
        ssh_url = None
        ssh_password = None


    # start the workers
    workers_v = {}
    log.info("starting the workers")
    for flowdir in ['dual']:
        pworker = start_daemon_ublarflow_worker(broker_address,flowdir,
                                                weights_files[flowdir],
                                                device,args.batch_size,False,False,
                                                ssh_url,ssh_password)
        workers_v[flowdir] = pworker
        
    print "[CTRL+C] to quit."
    log.info("Workers started")
    nalive = 1
    while nalive==1:
        time.sleep(10)
        nalive = 0
        for flowdir in ['dual']:
            if workers_v[flowdir].is_alive():
                nalive+=1
    log.info("At least one worker stopped")
        
