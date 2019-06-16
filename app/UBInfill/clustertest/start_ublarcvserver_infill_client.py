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
parser.add_argument("-c","--chstatus",type=str,default="wire",
                    help="chstatus producer from input")
parser.add_argument("-n","--infill",type=str,default="infill",
                    help="output tree name")
parser.add_argument("-t","--tick",type=bool,default=True,
                    help="specifies whether tick backwards")


if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])

    from ublarcvserver import Broker
    from UBInfillClient import UBInfillClient

    endpoint = args.brokeraddr

    level = logging.DEBUG
    if args.debug:
        level = logging.DEBUG

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile,level=level)

    log = logging.getLogger("start_ublarcvsever_worker_main")
    logging.basicConfig(level=logging.DEBUG)

    client = UBInfillClient(args.brokeraddr,args.input,args.output,
                            args.adc,args.chstatus,args.tick, args.infill)
    client.connect()

    #client.process_entry(0,args.tick)
    client.process_entries(args.tick)

    print("processed")

    client.finalize()
