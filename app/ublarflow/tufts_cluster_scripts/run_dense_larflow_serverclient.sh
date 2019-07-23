#!/bin/bash

# assume we are in the container

BROKER=$1

SUPERA_FILE=$2
TREENAME=$3

OUTPUT_DIR=$4
OUTPUT_FILE=$5

WORKDIR=$6
UBDL_DIR=$7
LARFLOW_DIR=$8

# setup container
echo "UBDL_DIR: ${UBDL_DIR}"
source ${UBDL_DIR}/setenv.sh
source ${UBDL_DIR}/configure.sh

cd ${UBDL_DIR}/ublarcvserver
source configure.sh

# add the deploy folder to the python path and path
export PYTHONPATH=${LARFLOW_DIR}:${PYTHONPATH}
export PATH=${LARFLOW_DIR}:${PATH}
echo "PATH: ${PATH}"

cd ${WORKDIR}

outfile_local=output_larflow.root
run_dense_ublarflow_client.py -i ${SUPERA_FILE} -o ${outfile_local} -n ${TREENAME} -l ${WORKDIR}/client.log -d ${BROKER}

mkdir -p ${OUTPUT_DIR}

# copy the file
scp ${outfile_local} ${OUTPUT_DIR}/${OUTPUT_FILE}