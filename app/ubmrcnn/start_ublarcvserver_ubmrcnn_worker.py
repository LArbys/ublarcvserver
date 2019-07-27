#!/usr/bin/env python
import os,sys,argparse,logging,time
import getpass

parser = argparse.ArgumentParser()
parser.add_argument("brokeraddr",type=str,help="Broker Address")
parser.add_argument("-l","--logfile",type=str, default=None,
                    help="where the log file is writen to")
parser.add_argument("-d","--debug",action="store_true",
                    help="set logger level to debug")
parser.add_argument("-w","--weights-dir",type=str,default=None,
                    help="directory where weights can be found")
parser.add_argument("-m","--mode",type=str,default="0",
                    help="run with device # (ubmrcnn requires cuda)")
parser.add_argument("-b","--batch-size",type=int,default=1,
                    help="batch size for each worker")
parser.add_argument("-n","--num_workers",type=int,default=1,
                    help="number of workers to launch")
parser.add_argument("-t","--ssh-tunnel",type=str,default=None,
                    help="Tunnel using SSH through the given IP address")
parser.add_argument("-u","--ssh-user",type=str,default=None,
                    help="username for ssh tunnel command")
parser.add_argument("-p","--plane",type=int,default=None,
                    help="set single plane to run on")


if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])
    from ublarcvserver import Broker
    from start_ubmrcnn_worker import startup_ubmrcnn_workers

    #set up CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.mode

    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("start_ublarcvsever_worker_main")
    logging.basicConfig(level=logging.INFO)

    weights_dir = "../ublarcvserver/networks/ubmrcnn/"
    if args.weights_dir is not None:
        weights_dir = args.weights_dir
    weights_files = {0:weights_dir+"/mcc8_mrcnn_plane0.pth",
                     1:weights_dir+"/mcc8_mrcnn_plane1.pth",
                     2:weights_dir+"/mcc8_mrcnn_plane2.pth"}

    for p in range(2):
        if not os.path.exists(weights_files[p]):
            log.error("did not find weight file at: "+weights_files[p])
            sys.exit(1)

    endpoint   = args.brokeraddr
    if args.batch_size !=1:
        log.info("batch size in MaskRCNN not tested beyond 1.")
    batch_size = args.batch_size
    nworkers   = args.num_workers
    endpoint = args.brokeraddr

    if args.ssh_tunnel is not None:
        if args.ssh_user is None:
            raise ValueError("If using ssh tunnel, must provide user")
        print("Using ssh, please provide password")
        ssh_password =  getpass.getpass()
        ssh_url = "%s@%s"%(args.ssh_user,args.ssh_tunnel)
    else:
        ssh_url = None
        ssh_password = None

    workers_v = []
    log.info("starting the workers")
    for w in range(args.num_workers):
        planes = [0,1,2]
        if args.plane is not None:
            planes = [args.plane]
        pworkers = startup_ubmrcnn_workers(endpoint,weights_files,
                                           device_id=args.mode,
                                           nplanes=planes,
                                           batch_size=batch_size,
                                           ssh_thru_server=ssh_url,
                                           ssh_password=ssh_password)


        workers_v.append(pworkers)

    log.info("Workers started")
    nalive = len(workers_v)
    while nalive>0:
        time.sleep(10)
        nalive = 0
        for plane_v in workers_v:
            for w in plane_v:
                if w.is_alive():
                    nalive+=1
    log.info("All workers stopped")
