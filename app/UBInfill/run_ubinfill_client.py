import os,sys,time,logging,argparse
from ublarcvserver import start_broker
from start_infill_worker import startup_infill_workers
from UBInfillClient import UBInfillClient


if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"

    # change to my directory, get rid of opreco? - means changing Client constructer
    input_dir = "/mnt/disk1/nutufts/kmason/ubdl/testdata/infill"
    larcv_supera_file   = input_dir+"/supera-larcv2-Run000001-SubRun006867.root"
    output_larcv_filename = "out_infill_test.root"

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworkers = startup_infill_workers(endpoint, devices="cuda" ,nplanes=[0,1,2],
                                        batch_size=1)

    # tick_backwards = true for larcv, false for larcv2 input
    client = UBInfillClient(endpoint,larcv_supera_file,"wire",
                            output_larcv_filename, tick_backwards=True)
    client.connect()

    client.process_entry(1)

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
