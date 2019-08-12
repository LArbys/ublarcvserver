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
parser.add_argument("-t","--tick",type=bool,default=True,
                    help="specifies whether tick backwards")
parser.add_argument("--local",action="store_true",default=False,
                    help="runs a local job with a broker and worker on an inter process socket (ipc)")
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
        weights_files = {0:args.weights_dir+"/Plane0Weights-13999.ckpt",
                         1:args.weights_dir+"/Plane1Weights-17999.ckpt",
                         2:args.weights_dir+"/Plane2Weights-26999.ckpt"}
                
        workers_v = startup_sparse_uresnet_workers(endpoint,
                                                   weights_files,
                                                   nplanes=[0,1,2],
                                                   device_id="cpu",
                                                   batch_size=1)
    

    client = SparseSSNetClient(args.brokeraddr,args.input,args.output,
                               adc_producer=args.adc,
                               input_mode=SparseSSNetClient.SPLIT,
                               tick_backwards=args.tick )
    client.connect()

    client.process_entries(start=0,end=1)

    print("processed")

    client.finalize()
