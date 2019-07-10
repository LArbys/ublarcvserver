#!/bin/bash

BROKER=$1

WORKDIR=$2

UBMRCNN_DIR=$3
UBDL_DIR=$4

INPUT_FILE=$5
OUTPUT_FILE=$6

# setup dependencies
cd ${UBDL_DIR}
source setenv.sh
export LARCV_OPENCV=0
source configure.sh
cd ${UBDL_DIR}/ublarcvserver
source configure.sh

export PYTHONPATH=${UBMRCNN_DIR}:${PYTHONPATH}
export PATH=${UBMRCNN_DIR}:${PATH}

cd ${WORKDIR}

logfile=client.log
script=start_ublarcvserver_ubmrcnn_client.py

export >& $logfile

cmd="${script} -l ${logfile} -d ${BROKER} -i ${INPUT_FILE} -o ${OUTPUT_FILE} -t True"

echo $cmd
$cmd
