#!/bin/bash

cd /cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl
source setenv.sh
source configure.sh
cd scripts
source setenv_ublarcvserver.sh

cd ublarcvserver
source configure.sh
cd app/UBInfill/clustertest
pwd

weightdir=/cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl/ublarcvserver/networks/infill
let line=${SLURM_ARRAY_TASK_ID}+1
devnum=`sed -n ${line}p /cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl/ublarcvserver/app/UBInfill/clustertest/tufts_pgpu03_assignments.txt`
device=`printf cuda:%d ${devnum}`
logfile=`printf /tmp/worker_id%d.log ${SLURM_ARRAY_TASK_ID}`
script=/cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl/ublarcvserver/app/UBInfill/clustertest/start_ublarcvserver_infill_worker.py
#scrpit=start_ublarcvserver_worker.py
echo ${device}
cmd="python ${script} -l ${logfile} -d tcp://localhost:6000 -m ${device} -w ${weightdir}"

echo $cmd
$cmd
