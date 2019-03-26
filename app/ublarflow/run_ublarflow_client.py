import os,sys,time,logging,argparse
from ublarcvserver import start_broker
from start_ublarflow_worker import start_daemon_ublarflow_worker
from UBSparseLArFlowClient import UBSparseLArFlowClient


if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "../../networks/larflow/weights/"
    weights_files = {2:weights_dir+"/checkpoint.500th.tar"}

    #supera_larcv_filename = "/home/twongjirad/working/larbys/ubdl/testdata/mcc9jan_extbnb/supera-Run005121-SubRun000004.root"
    supera_larcv_filename = "./../../../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    output_larcv_filename = "out_ublarflow_test.root"

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworker = start_daemon_ublarflow_worker(endpoint,2,weights_files[2],
                                            "cuda",1,None,None)

    client = UBSparseLArFlowClient(endpoint,
                                   supera_larcv_filename,
                                   output_larcv_filename,
                                   adc_producer="wiremc",
                                   save_as_sparseimg=True,
                                   tick_backwards=True)

    client.connect()

    client.process_entry(0)

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
