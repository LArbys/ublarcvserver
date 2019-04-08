#!/usr/bin/env python
import os,sys,time,logging,argparse

"""
Provides script to start worker for Dense LArFlow Network.

Also provides utility functions for starting worker(s).
"""

parser = argparse.ArgumentParser()
# required
parser.add_argument("brokeraddr",type=str,help="Broker Address")
parser.add_argument("-i","--input", required=True,type=str,help="Input Supera File")
parser.add_argument("-o","--output",required=True,type=str,help="Output Supera File")
parser.add_argument("-n","--name",required=True,type=str,
                    help="Producer/Tree name with images to process. e.g. wire or wiremc")

# options
parser.add_argument("-l","--logfile",type=str, default=None,
                    help="where the log file is writen to")
parser.add_argument("-d","--debug",action="store_true",
                    help="set logger level to debug")
parser.add_argument("-t","--ssh-tunnel",type=str,default=None,
                    help="Tunnel using SSH through the given IP address")
parser.add_argument("-u","--ssh-user",type=str,default=None,
                    help="username for ssh tunnel command")
parser.add_argument("-s","--self-contained",action="store_true",default=False,
                    help="Self-contained mode. Launches broker and workers. Useful for running/testing locally.")
parser.add_argument("-w","--weights-dir",type=str,default=None,
                    help="directory where weights can be found. only for self-contained mode.")
parser.add_argument("-m","--device",type=str,default="cuda",
                    help="run with device. either 'cuda' or 'cpu'. only for self-contained mode.")
parser.add_argument("-b","--batch-size",type=int,default=1,
                    help="batch size for each worker. only for self-contained mode.")


args = parser.parse_args(sys.argv[1:])
from UBDenseLArFlowClient import UBDenseLArFlowClient


if __name__ == "__main__":

    # key parameters
    endpoint  = args.brokeraddr

    # example input/output
    #supera_larcv_filename = "./../../../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    #output_larcv_filename = "out_dense_ublarflow_test.root"

    supera_larcv_filename = args.input
    output_larcv_filename = args.output

    if os.path.exists(output_larcv_filename):
        print "Output file already exists. Given path=",output_larcv_filename
        sys.exit(0)

    if not os.path.exists(supera_larcv_filename):
        print "Input file could not be found. Given path=",supera_larcv_filename
        sys.exit(0)

    # configure logger
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)
    else:
        logging.basicConfig(level=level)

    log = logging.getLogger("run_dense_ublarflow_client:main")
    
    
    workers = []
    pbroker  = None
    if args.self_contained:
        # self-contained mode. We start broker and workers to run
        from ublarcvserver import start_broker
        from start_dense_ublarflow_worker import start_daemon_ublarflow_worker        

        if args.weights_dir is None:
            print "In self-contained mode, must provide weights_dir argument (-w) to load worker networks"
            
        weights_dir = args.weights_dir
        weights_files = {"y2u":weights_dir+"/devfiltered_larflow_y2u_832x512_32inplanes.tar",
                         "y2v":weights_dir+"/devfiltered_larflow_y2v_832x512_32inplanes.tar"}

        
        bindpoint = "tcp://*:%d"%(int(endpoint.split(":")[-1]))
        pbroker = start_broker(bindpoint)
        use_half = False
        pworker_y2u = start_daemon_ublarflow_worker(endpoint,'y2u',weights_files['y2u'],
                                                    args.device,1,use_half,None,None)
        pworker_y2v = start_daemon_ublarflow_worker(endpoint,'y2v',weights_files['y2v'],
                                                    args.device,1,use_half,None,None)
        workers.append(pworker_y2u)
        workers.append(pworker_y2v)

    # get the ssh url and password if specified in arguments
    if args.ssh_tunnel is not None:
        import getpass
        if args.ssh_user is None:
            raise ValueError("If using ssh tunnel, must provide user")
        print "Using ssh, please provide password"
        ssh_password =  getpass.getpass()
        ssh_url = "%s@%s"%(args.ssh_user,args.ssh_tunnel)
    else:
        ssh_url = None
        ssh_password = None
    
    log.info("Starting client to process %s"%(supera_larcv_filename))
    log.info("connecting to {}".format(endpoint))
    client = UBDenseLArFlowClient(endpoint,
                                  supera_larcv_filename,
                                  output_larcv_filename,
                                  adc_producer=args.name,
                                  flow_dirs=["y2u","y2v"],
                                  tick_backwards=True,
                                  ssh_thru_server=ssh_url,
                                  ssh_password=ssh_password)
                                  

    client.connect()

    client.process_entries()

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
