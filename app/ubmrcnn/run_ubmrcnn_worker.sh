#!/bin/bash
basedir=$PWD
#cd /home/twongj01/working/larbys/py3_ubdl
cd $UBDL_BASEDIR

# alias python='python3'
source setenv.sh
source configure.sh
export LARCV_OPENCV=0
source setenv.sh
source configure.sh
cd scripts
source setenv_ublarcvserver.sh
SLURM_ARRAY_TASK_ID=0
weightdir=/home/jmills/workdir/ubdl/ublarcvserver/app/ubmrcnn
let line=${SLURM_ARRAY_TASK_ID}+1
# devnum=`sed -n ${line}p /cluster/tufts/wongjiradlab/larbys/pubs/dlleepubs/ubdlserver/tufts_pgpu03_assignments.txt`
#on meitner must set dev manually
# devnum is either "cuda:#" or "cpu"
devnum="cpu"
# device=`printf cuda:%s ${devnum}`
logfile=`printf /tmp/worker_id%d.log ${SLURM_ARRAY_TASK_ID}`
script=/home/jmills/workdir/ubdl/ublarcvserver/app/ubmrcnn/start_ublarcvserver_ubmrcnn_worker.py
# scrpit=start_ublarcvserver_worker.py
cd $basedir
logfile=loggpu.log

cmd="python3 ${script} -l ${logfile} -d tcp://10.246.81.72:6000  -m ${devnum} -w ${weightdir} -t  xfer.cluster.tufts.edu  -u jmills09"

$cmd
