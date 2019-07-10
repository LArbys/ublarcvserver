#!/usr/bin/env python
import os,sys,argparse,logging,time
import getpass

parser = argparse.ArgumentParser()
parser.add_argument("-brokeraddr",type=str,help="Broker Address")
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
parser.add_argument("-c","--chstatus",type=str,default="wire",
                    help="chstatus producer from input")
parser.add_argument("-n","--infill",type=str,default="infill",
                    help="output tree name")
parser.add_argument("-t","--tick",type=bool,default=True,
                    help="specifies whether tick backwards")
parser.add_argument("--local",action="store_true",default=False,
                    help="runs a local job with a broker and worker on an inter process socket (ipc)")
parser.add_argument("--weights-dir",type=str,default="None",
                    help="specify path to directory with weights (assumes weight names)")

if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])

    from UBInfillSparseClient import UBInfillSparseClient

    endpoint = args.brokeraddr

    level = logging.DEBUG
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("start_ublarcvsever_worker_main")
    logging.basicConfig(level=logging.DEBUG)

    workers_v = None
    broker    = None
    if args.local:
        from ublarcvserver.start_broker import start_broker
        from start_sparseinfill_worker import startup_infill_workers

        if len(endpoint)<5 or endpoint[:6]!="ipc://":
            raise ValueError("Broker address must be an IPC socker. Addresses look like: 'ipc:///tmp/feeds/mysocketname'. Got: {}".format(endpoint))
        # start a broker daemon
        broker = start_broker(endpoint)

        # weight files
        weights_files = {0:args.weights_dir+"/sparseinfill_uplane_test.tar",
                         1:args.weights_dir+"/sparseinfill_vplane_test.tar",
                         2:args.weights_dir+"/sparseinfill_yplane_test.tar"}
        
        workers_v = startup_infill_workers(endpoint,
                                           weights_files,
                                           nplanes=[0,1,2],
                                           devices="cpu",
                                           batch_size=1)        

    client = UBInfillSparseClient(args.brokeraddr,args.input,args.output,
                                  adc_producer=args.adc,
                                  chstatus_producer=args.chstatus,
                                  tick_backwards=args.tick,
                                  infill_tree_name=args.infill)
    client.connect()

    #client.process_entry(0,args.tick)
    client.process_entries()

    print("processed")

    client.finalize()
