#!/bin/bash

# assumes environment already set
# needs following scripts to set environment
# must be called in ublarcvserver/app/ublarflow

# ubdl/setenv.sh
# ubdl/configure.sh
# ublarcvserver/configure.sh
# ublarcvserver/networks/larflow/setenv.sh
python start_dense_ublarflow_worker.py -d -l worker_00.log -w ../../networks/larflow/weights/ -m "cuda:0" -b 1 -t fastx-dev.cluster.tufts.edu -u twongj01 tcp://10.246.81.72:6000
