#!/bin/bash

cd /cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl
source setenv.sh
source configure.sh
cd scripts
source setenv_ublarcvserver.sh

cd ublarcvserver
source configure.sh
cd app/UBInfill/clustertest
`
logfile=infill_client.log
script=start_ublarcvserver_infill_client.py
#scrpit=start_ublarcvserver_worker.py
inputfile=testfile.root
outputfile=infill_output.root
cmd="python ${script} -l ${logfile} -d tcp://localhost:6000 -i ${inputfile} -o ${outputfile} -t True"


echo $cmd
$cmd
