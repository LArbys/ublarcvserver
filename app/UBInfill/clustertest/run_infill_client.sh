#!/bin/bash

cd /cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl
source setenv.sh
source configure.sh

cd ublarcvserver
source configure.sh
cd app/UBInfill/clustertest

logfile=infill_client.log
script=start_ublarcvserver_infill_client.py
#scrpit=start_ublarcvserver_worker.py
inputfile=testfilemcc8.root
outputfile=infill_outputmcc8.root
cmd="python ${script} -l ${logfile} -d tcp://10.246.81.72:6000 -i ${inputfile} -o ${outputfile} -t False"


echo $cmd
$cmd
