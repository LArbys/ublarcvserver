#!/bin/bash

cd /usr/local/ubdl
source setenv.sh
source configure.sh
export LARCV_OPENCV=0
source setenv.sh
source configure.sh
cd scripts
source setenv_ublarcvserver.sh

weightdir=/media/disk1/jmills/cluster_weights
let line=${SLURM_ARRAY_TASK_ID}+1
devnum=`sed -n ${line}p /cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/ubdlserver/tufts_pgpu03_assignments.txt`
device=`printf cuda:%d ${devnum}`
logfile=`printf /tmp/worker_id%d.log ${SLURM_ARRAY_TASK_ID}`
script=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/ubdlserver/start_ublarcvserver_mrcnn_worker.py
#scrpit=start_ublarcvserver_worker.py
cmd="python ${script} -l ${logfile} -d tcp://localhost:6000 -m ${device} -w ${weightdir}"

echo $cmd
$cmd
