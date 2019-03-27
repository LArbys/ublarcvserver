import os,sys,time,logging,argparse
from ublarcvserver import start_broker
from start_dense_ublarflow_worker import start_daemon_ublarflow_worker
from UBDenseLArFlowClient import UBDenseLArFlowClient


if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "../../networks/larflow/weights/"
    weights_files = {"y2u":weights_dir+"/devfiltered_larflow_y2u_832x512_32inplanes.tar",
                     "y2v":weights_dir+"/devfiltered_larflow_y2v_832x512_32inplanes.tar"}

    #supera_larcv_filename = "/home/twongjirad/working/larbys/ubdl/testdata/mcc9jan_extbnb/supera-Run005121-SubRun000004.root"
    supera_larcv_filename = "./../../../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    output_larcv_filename = "out_dense_ublarflow_test.root"

    logging.basicConfig(level=logging.DEBUG)

    pbroker = start_broker(bindpoint)
    pworker_y2u = start_daemon_ublarflow_worker(endpoint,'y2u',weights_files['y2u'],
                                            "cuda",1,None,None)
    pworker_y2v = start_daemon_ublarflow_worker(endpoint,'y2v',weights_files['y2v'],
                                            "cuda",1,None,None)

    client = UBDenseLArFlowClient(endpoint,
                                  supera_larcv_filename,
                                  output_larcv_filename,
                                  adc_producer="wiremc",
                                  tick_backwards=True)

    client.connect()

    client.process_entry(0)

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
