#!/bin/bash

# get location called
__run_ubmrcnn_worker_start_dir__=`pwd`

# location script is run
__run_ubmrcnn_worker_where__="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# broker
broker=$1
# location of ubmrcnn script so run
ubmrcnn_dir=$2
# location of ubdl for dependencies
ubdl_dir=$3
# location of weights
weight_dir=$4
# path to text file listing gpu assignments
gpu_list=$5
# log file
log_file=$6

# setup dependencies
cd ${ubdl_dir}
source setenv.sh
export LARCV_OPENCV=0
source configure.sh
cd ${ubdl_dir}/ublarcvserver
source configure.sh

# setup the script dir
cd ${ubmrcnn_dir}
source setenv.sh

let line=${SLURM_ARRAY_TASK_ID}+1
#devnum=`sed -n ${line}p ${gpu_list}`
#device=`printf cuda:%d ${devnum}`
device="cpu"
#device="cuda"
echo "SETUP WOKER TO RUN ON DEVICE=${device}"
script=${ubmrcnn_dir}/start_ublarcvserver_ubmrcnn_worker.py

export CUDA_VISIBLE_DEVICES=${devnum}

cmd="python ${script} -l ${log_file} -d -m ${device} -w ${weight_dir} ${broker}"

echo $cmd
$cmd
