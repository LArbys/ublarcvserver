#!/bin/bash

inputfile=$1
outputfile=$2
workdir=$3
logdir=$4

PORT=6000
SERVER_ADDRESS="tcp://10.246.81.72"
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl

# assume that shell is configured by
# ubdl/setenv.sh
# ubdl/configure.sh
# ublarcvserver/configure.sh
# ublarcvserver/app/ublarflow/setenv.sh

cd $UBDL_DIR
source setenv.sh
source configure.sh
cd ublarcvserver
source configure.sh
cd app/ublarflow
source setenv.sh

cd $workdir
run_dense_ublarflow_client.py -d -l $logdir -n wire -i $inputfile -o $outputfile ${SERVER_ADDRESS}:${PORT}
