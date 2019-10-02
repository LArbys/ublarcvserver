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


def start_sparse_uresnet_worker(broker_address,plane,
                                weight_file,
                                device_id,
                                batch_size,
                                nrows,
                                ncols,
                                nlayers,
                                nfilters,
                                ssh_thru_server,
                                ssh_password):

    from SparseSSNetWorker import SparseSSNetWorker
    
    worker=SparseSSNetWorker(broker_address,plane,weight_file,
                             batch_size,
                             row_tick_dim=nrows,
                             col_wire_dim=ncols,
                             device_id=device_id,
                             nlayers=nlayers,
                             nfilters=nfilters,
                             ssh_thru_server=ssh_thru_server,
                             ssh_password=ssh_password )
    worker.connect()
    print("sparse uresnet worker started: ",worker.idname())
    worker.run()


def startup_sparse_uresnet_workers( broker_address, weights_files,
                                    nrows=512, ncols=512,
                                    device_id=0,
                                    batch_size=1,
                                    nplanes=[0,1,2],
                                    nlayers=5,
                                    nfilters=32,
                                    ssh_thru_server=None, ssh_password=None,
                                    start=True):

    from multiprocessing import Process
    
    # setup the worker
    pworkers = []
    for p in nplanes:
        pworker = Process(target=start_sparse_uresnet_worker,
                          args=(broker_address,p,weights_files[p],
                                device_id,batch_size,nrows,ncols,
                                nlayers, nfilters,
                                ssh_thru_server,ssh_password))
        pworker.daemon = True
        pworkers.append(pworker)

    if start:
        for pworker in pworkers:
            pworker.start()

    return pworkers


if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])
    from ublarcvserver import Broker

    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("start_sparse_uresnet_worker_main")
    logging.basicConfig(level=logging.INFO)

    weights_dir = "./weights"
    if args.weights_dir is not None:
        weights_dir = args.weights_dir
    weights_files = {0:weights_dir+"/Plane0Weights-13999.ckpt",
                     1:weights_dir+"/Plane1Weights-17999.ckpt",
                     2:weights_dir+"/Plane2Weights-26999.ckpt"}

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
        pworkers = startup_sparse_uresnet_workers(endpoint,weights_files,
                                                  nrows=512,ncols=512,
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
