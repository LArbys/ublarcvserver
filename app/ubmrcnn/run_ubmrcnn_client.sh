#!/bin/bash
basedir=$PWD
cd /cluster/tufts/wongjiradlab/jmills09/ubdl/
# alias python='python3'
source setenv.sh
source configure.sh
export LARCV_OPENCV=0
source setenv.sh
source configure.sh

cd ublarcvserver
source configure.sh
cd app/ubmrcnn/

logfile=client.log
script=start_ublarcvserver_ubmrcnn_client.py

inputfile=testfile.root
outputfile=ubmrcnn_output.root
cmd="python ${script} -l ${logfile} -d tcp://10.246.81.72:6000 -i ${inputfile} -o ${outputfile} -t True"


echo $cmd
$cmd
