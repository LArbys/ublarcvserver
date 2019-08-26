#!/usr/bin/env python
from __future__ import print_function
import os,sys,argparse,logging,time
import getpass

parser = argparse.ArgumentParser()
parser.add_argument("brokeraddr",type=str,help="Broker Address")
parser.add_argument("-l","--logfile",type=str, default=None,
                    help="where the log file is writen to")
parser.add_argument("-d","--debug",action="store_true",
                    help="set logger level to debug")
parser.add_argument("-i","--input",type=str,default=None,
                    help="input file to run over")
parser.add_argument("-o","--output",type=str,default=None,
                    help="output file name")
parser.add_argument("-a","--adc",type=str,default="wire",
                    help="adc producer from input")
parser.add_argument("-n","--out_tree_name",type=str,default="mrcnn",
                    help="output tree name")
parser.add_argument("-tb","--tickbackwards",action="store_true",default=False,
                    help="specifies whether tick backwards")
parser.add_argument("-sm","--split-mode",type=int,default=0,
                    help="split mode: 0=whole, 1=split, 2=opflash")
parser.add_argument("--local",action="store_true",default=False,
                    help="runs a local job with a broker and worker on an inter process socket (ipc)")
parser.add_argument("--nrows",type=int,default=512,
                    help="number of rows of image. needed if starting worker")
parser.add_argument("--ncols",type=int,default=512,
                    help="number of columns of image. needed if starting worker")
parser.add_argument("--weights-dir",type=str,default="None",
                    help="specify path to directory with weights (assumes weight names)")


if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])

    from SparseSSNetClient import SparseSSNetClient

    endpoint = args.brokeraddr

    level = logging.DEBUG
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("start_sparse_uresnet_client_main")
    logging.basicConfig(level=logging.DEBUG)

    if args.split_mode not in [SparseSSNetClient.WHOLE,
                               SparseSSNetClient.SPLIT,
                               SparseSSNetClient.OPFLASH_ROI]:
        print("unrecognized image pre-processing mode: {}".format(args.split_mode))
        sys.exit(1)

    workers_v = None
    broker    = None
    if args.local:
        from ublarcvserver.start_broker import start_broker
        from start_sparse_uresnet_worker import startup_sparse_uresnet_workers

        if len(endpoint)<5 or endpoint[:6]!="ipc://":
            raise ValueError("Broker address must be an IPC socker. Addresses look like: 'ipc:///tmp/feeds/mysocketname'. Got: {}".format(endpoint))
        # start a broker daemon
        broker = start_broker(endpoint)

        # weight files
        #weights_files = {0:args.weights_dir+"/Plane0Weights-13999.ckpt",
        #                 1:args.weights_dir+"/Plane1Weights-17999.ckpt",
        #                 2:args.weights_dir+"/Plane2Weights-26999.ckpt"}
        weights_files = {0:args.weights_dir+"/snapshot_plane0_16filter_6layers_22999.ckpt",
                         1:args.weights_dir+"/snapshot_plane1_16filter_6layers_31999.ckpt",
                         2:args.weights_dir+"/snapshot_plane2_16filter_6layers_36999.ckpt"}
        #weights_files = {0:args.weights_dir+"/snapshot_32filters_5layers-15999.ckpt",
        #                 1:args.weights_dir+"/snapshot_32filters_5layers-15999.ckpt",
        #                 2:args.weights_dir+"/snapshot_32filters_5layers-15999.ckpt"}
                
        workers_v = startup_sparse_uresnet_workers(endpoint,
                                                   weights_files,
                                                   nrows=args.nrows,
                                                   ncols=args.ncols,
                                                   nplanes=[0,1,2],
                                                   device_id="cpu",
                                                   batch_size=1)
    
    
    client = SparseSSNetClient(args.brokeraddr,
                               args.input,args.output,
                               adc_producer=args.adc,
                               input_mode=args.split_mode,
                               planes=[0,1,2],
                               tick_backwards=args.tickbackwards )
    client.connect()

    client.process_entries(start=0,end=0)
    #client.process_entries()

    print("processed")

    client.finalize()
