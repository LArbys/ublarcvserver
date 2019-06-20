#!/bin/bash

cd /cluster/tufts/wongjiradlab/kmason03/testdir/ubdl
source setenv.sh
source configure.sh
cd scripts
source setenv_ublarcvserver.sh

cd ../ublarcvserver
source configure.sh
cd app/UBInfill/sparseinfill
pwd

weightdir=/cluster/tufts/wongjiradlab/larbys/ssnet_models/
let line=${SLURM_ARRAY_TASK_ID}+1
devnum=`sed -n ${line}p /cluster/tufts/wongjiradlab/kmason03/testdir/ubdl/ublarcvserver/app/UBInfill/sparseinfill/tufts_pgpu03_assignments.txt`
device=`printf cuda:%d ${devnum}`
logfile=`printf /tmp/worker_id%d.log ${SLURM_ARRAY_TASK_ID}`
script=/cluster/tufts/wongjiradlab/kmason03/testdir/ubdl/ublarcvserver/app/UBInfill/sparseinfill/start_ublarcvserver_sparseinfill_worker.py
echo ${device}
cmd="python ${script} -l ${logfile} -brokeraddr tcp://localhost:6000 -m ${device} -w ${weightdir}"

echo $cmd
$cmd
