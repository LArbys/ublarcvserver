import os,sys,time,logging
from start_broker_worker import startup_broker_and_workers
from UBSSNetClient import UBSSNetClient

if __name__ == "__main__":

    # endpoint:
    endpoint  = "tcp://localhost:6005"
    bindpoint = "tcp://*:6005"
    weights_dir = "/home/twongjirad/working/nutufts/pytorch-uresnet/weights/"
    weights_files = {2:weights_dir+"/saved_caffe_weights_plane2.tar"}

    input_dir = "/home/twongjirad/working/larbys/ubdl/testdata/ex1"
    larcv_supera_file   = input_dir+"/supera-Run000001-SubRun006867.root"
    larlite_opreco_file = input_dir+"/opreco-Run000001-SubRun006867.root"
    output_larcv_filename = "out_ubssnet_test.root"

    logging.basicConfig(level=logging.DEBUG)

    pbroker, pworkers = startup_broker_and_workers(endpoint,bindpoint,
                                                    weights_files,nplanes=[2])


    client = UBSSNetClient(endpoint,larcv_supera_file,"wire",
                            output_larcv_filename,
                            larlite_opreco_file=larlite_opreco_file,
                            apply_opflash_roi=True, tick_backwards=True)
    client.connect()

    client.process_entry(1)

    client.finalize()

    print "[ENTER] to quit."
    raw_input()
