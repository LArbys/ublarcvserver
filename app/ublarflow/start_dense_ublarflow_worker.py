#!/usr/bin/env python
import os,sys,logging,time,argparse
import getpass

"""
Provides script to start worker for Dense LArFlow Network.

Also provides utility functions for starting worker(s).
"""

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

args = parser.parse_args(sys.argv[1:])

from multiprocessing import Process
from UBDenseLArFlowWorker import UBDenseLArFlowWorker
from ublarcvserver import start_broker


def start_dense_ublarflow_worker(broker_address,flow_dir,weight_file,
                                 device,batch_size,use_half,
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
    ssh_thru_server str Address of machine we should connect to via ssh
    ssh_password std Unprotected password to the machine
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

    """
    start single copy of dense larflow worker on a new process
    Note: arguments are the same as above function. Just passes through.

    return Process object in daemon mode running dense larflow worker object
    """
    
    pworker = Process(target=start_dense_ublarflow_worker,
                      args=(broker_address,flow_dir,weight_file,
                            device,batch_size,use_half,
                            ssh_thru_server,ssh_password))
    pworker.daemon = True
    pworker.start()
    return pworker



if __name__ == "__main__":

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
    log = logging.getLogger("start_denselarflow_worker_main")

    # set weight directionry
    weights_dir = "../ublarcvserver/networks/larflow/weights/"
    if args.weights_dir is not None:
        weights_dir = args.weights_dir
    weights_files = {"y2u":weights_dir+"/devfiltered_larflow_y2u_832x512_32inplanes.tar",
                     "y2v":weights_dir+"/devfiltered_larflow_y2v_832x512_32inplanes.tar"}
    
    # set the device
    device = args.device

    # broker address
    broker_address = args.brokeraddr
    if "tcp://" not in broker_address:
        broker_address = "tcp://"+broker_address

    # if specified in arguments. start a broker
    if args.launch_broker:
        pbroker = start_broker(broker_address)
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
    for flowdir in ['y2u','y2v']:
        pworker = start_daemon_ublarflow_worker(broker_address,flowdir,
                                                weights_files[flowdir],
                                                device,args.batch_size,False,
                                                ssh_url,ssh_password)
        workers_v[flowdir] = pworker
        
    print "[CTRL+C] to quit."
    log.info("Workers started")
    nalive = 2
    while nalive==2:
        time.sleep(60)
        nalive = 0
        for flowdir in ['y2u','y2v']:
            if workers_v[flowdir].is_alive():
                nalive+=1
    log.info("At least one worker stopped")
        
