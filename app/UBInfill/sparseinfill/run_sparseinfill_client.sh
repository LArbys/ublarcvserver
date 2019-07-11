#!/bin/bash

# assume we are in the container

# location of UBDL code to use
WORKDIR=$1
SUPERA_FILE=$2
OUTPUT_DIR=$3
OUTPUT_FILE=$4
UBDL_DIR=$5
INFILL_DIR=$6
WEIGHT_DIR=$7

# setup container
echo "UBDL_DIR: ${UBDL_DIR}"
source ${UBDL_DIR}/setenv.sh
source ${UBDL_DIR}/configure.sh
source ${UBDL_DIR}/ublarcvserver/configure.sh

# add the deploy folder to the python path and path
export PYTHONPATH=${INFILL_DIR}:${UBDL_DIR}/sparse_infill/sparse_infill/models:${PYTHONPATH}
export PATH=${INFILL_DIR}:${PATH}
echo "PATH: ${PATH}"

# make directory for IPC socket location
ipc_dir=/tmp/sparse_infill_${SLURM_JOB_ID}
mkdir -p ${ipc_dir}
ipc_addr="ipc://${ipc_dir}/client"

clientlog=client_${SLURM_JOB_ID}.log

# got to workdir
cd $WORKDIR

#output file
outfile_local=output_infill.root

start_ublarcvserver_sparseinfill_client.py -brokeraddr ${ipc_addr} -d -l ${clientlog} -i ${SUPERA_FILE} -o $outfile_local --local --weights-dir=${WEIGHT_DIR}

mkdir -p ${OUTPUT_DIR}

# copy the file
scp ${outfile_local} ${OUTPUT_DIR}/${OUTPUT_FILE}