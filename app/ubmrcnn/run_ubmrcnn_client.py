import os,sys,time,logging,argparse
from ublarcvserver import start_broker
from start_ubmrcnn_worker import startup_ubmrcnn_workers
from UBMRCNNClient import UBMRCNNClient
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "/media/disk1/jmills/cluster_weights"
    weights_files = {0:weights_dir+"/mcc8_mrcnn_plane0.pth",
                     1:weights_dir+"/mcc8_mrcnn_plane1.pth",
                     2:weights_dir+"/mcc8_mrcnn_plane2.pth"}

    input_dir = "/media/disk1/jmills/mcc9jan_extbnb_data"
    larcv_supera_file   = input_dir+"/larcv_wholeview_fffea264-968c-470e-9141-811b5f3d6dcd.root"
    # larlite_opreco_file = input_dir+"/opreco-Run000001-SubRun006867.root"
    output_larcv_filename = "out_mrcnn_test.root"

    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=None)

    pbroker = start_broker(bindpoint)

    pworkers = startup_ubmrcnn_workers(endpoint,weights_files,
                                        # devices="cuda",nplanes=[0,1,2],
                                        devices=["cuda", "cuda", "cuda"],nplanes=[0,1,2],
                                        batch_size=1)

    client = UBMRCNNClient(endpoint,larcv_supera_file,
                            output_larcv_filename,
                            "wire",
                            tick_backwards=True,
                            skip_detsplit=True)
    client.connect()

    client.process_entry(1)
    # client.process_entries()
    client.finalize()

    print("[ENTER] to quit.")
    if sys.version_info[0] < 3:
        raw_input()
    else:
        input()
