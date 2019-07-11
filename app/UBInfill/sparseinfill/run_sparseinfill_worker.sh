#!/bin/bash

broker=$1
UBDL_DIR=$2
INFILL_DIR=$3
weight_dir=$4
GPU_LIST=$5
logdir=$6

cd $UBDL_DIR
source setenv.sh
source configure.sh
cd ublarcvserver
source configure.sh

cd ${INFILL_DIR}/sparseinfill

let line=${SLURM_ARRAY_TASK_ID}+1
devnum=`sed -n ${line}p ${GPU_LIST}`
device=`printf cuda:%d ${devnum}`
logfile=`printf ${logdir}/infill_worker_id%d.log ${SLURM_ARRAY_TASK_ID}`
script=${INFILL_DIR}/sparseinfill/start_ublarcvserver_sparseinfill_worker.py
echo ${device}

device=cuda
export CUDA_VISIBLE_DEVICES=${devnum}
cmd="python ${script} -l ${logfile} -d -brokeraddr ${broker} -m ${device} -w ${weight_dir}"

echo $cmd
$cmd
