import os,sys,time,logging,argparse
from ublarcvserver import start_broker
from start_ubinfill_worker import startup_infill_workers
from UBInfillSparseClient import UBInfillSparseClient


if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"

    weights_dir = "/cluster/tufts/wongjiradlab/larbys/ssnet_models/"
    weights_files = {0:weights_dir+"/sparseinfill_uplane_test.tar",
                     1:weights_dir+"/sparseinfill_vplane_test.tar",
                     2:weights_dir+"/sparseinfill_yplane_test.tar"}


    # change to my directory, get rid of opreco? - means changing Client constructer
    input_dir = "/mnt/disk1/nutufts/kmason/ubdl/ublarcvserver/app/UBInfill/clustertest"
    larcv_supera_file   = input_dir+"/testfile.root"
    output_larcv_filename = "out_infill_test.root"

    logging.basicConfig(level=logging.INFO)

    pbroker = start_broker(bindpoint)
    pworkers = startup_infill_workers(endpoint,weights_files,
                                        devices="cuda:0" ,nplanes=[0,1,2],
                                        batch_size=1)

    # tick_backwards = true for larcv, false for larcv2 input
    client = UBInfillClient(endpoint,larcv_supera_file,
                            output_larcv_filename, "wire","wire",tick_backwards=False)
    client.connect()

    client.process_entry(0,False)

    client.finalize()

    print("[ENTER] to quit.")
    raw_input()
    print("FINISHED!!!!")
