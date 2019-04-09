#!/usr/bin/env python
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


if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])

    from ublarcvserver import Broker
    from UBMRCNNClient import UBMRCNNClient

    endpoint = args.brokeraddr

    level = logging.DEBUG
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("start_ublarcvsever_worker_main")
    logging.basicConfig(level=logging.DEBUG)

    client = UBMRCNNClient(args.brokeraddr,args.input,args.output,
                                adc_producer=args.adc, tick_backwards=args.tick, mrcnn_tree_name=args.out_tree_name)
    client.connect()

    client.process_entry(0)

    print("processed")

    client.finalize()
