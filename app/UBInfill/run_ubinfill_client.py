import os,sys,time,logging,argparse
from ublarcvserver import start_broker
from start_ubinfill_worker import startup_infill_workers
from UBInfillClient import UBInfillClient


if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"

    weights_dir = "/mnt/disk1/nutufts/kmason/ubdl/ublarcvserver/networks/infill/"
    # weights_files = {0:weights_dir+"/uplane_MC_40000.tar",
    #                  1:weights_dir+"/vplane_MC_21500.tar",
    #                  2:weights_dir+"/yplane_MC_33000.tar"}

    weights_files = {0:weights_dir+"/yplane_MC_33000.tar",
                  1:weights_dir+"/vplane_MC_21500.tar",
                  2:weights_dir+"/uplane_MC_40000.tar"}




    # change to my directory, get rid of opreco? - means changing Client constructer
    input_dir = "/mnt/disk1/nutufts/kmason/testdata"
    larcv_supera_file   = input_dir+"/larcv_wholeview_2051e379-6caa-45a3-838f-3dfc427533d5.root"
    output_larcv_filename = "out_infill_test.root"

    logging.basicConfig(level=logging.INFO)

    pbroker = start_broker(bindpoint)
    pworkers = startup_infill_workers(endpoint,weights_files,
                                        devices="cuda" ,nplanes=[0,1,2],
                                        batch_size=1)

    # tick_backwards = true for larcv, false for larcv2 input
    client = UBInfillClient(endpoint,larcv_supera_file,
                            output_larcv_filename, "wire","wire",tick_backwards=True)
    client.connect()

    client.process_entry(0)

    client.finalize()

    print("[ENTER] to quit.")
    raw_input()
    print("FINISHED!!!!")
