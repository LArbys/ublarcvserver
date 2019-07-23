#!/bin/bash

# assume we are in the container

# location of UBDL code to use
SUPERA_FILE=$1
OUTPUT_DIR=$2
OUTPUT_FILE=$3
UBDL_DIR=$4
WEIGHT_DIR=$5

# setup container
echo "UBDL_DIR: ${UBDL_DIR}"
source ${UBDL_DIR}/setenv.sh
source ${UBDL_DIR}/configure.sh

# add the deploy folder to the python path and path
LARFLOW_DEPLOY_DIR=${UBDL_DIR}/larflow/deploy
export PYTHONPATH=${LARFLOW_DEPLOY_DIR}:${PYTHONPATH}
export PATH=${LARFLOW_DEPLOY_DIR}:${PATH}
echo "PATH: ${PATH}"
ls ${LARFLOW_DEPLOY_DIR}

weights_y2u=${WEIGHT_DIR}/devfiltered_larflow_y2u_832x512_32inplanes.tar
outfile_local_y2u=output_larflow_y2u.root
run_larflow_wholeview.py -i ${SUPERA_FILE} -o ${outfile_local_y2u} -c ${weights_y2u} --flowdir y2u -g -1 -v -a

weights_y2v=${WEIGHT_DIR}/devfiltered_larflow_y2v_832x512_32inplanes.tar
outfile_local_y2v=output_larflow_y2v.root
run_larflow_wholeview.py -i ${SUPERA_FILE} -o ${outfile_local_y2v} -c ${weights_y2v} --flowdir y2v -g -1 -v -a

hadd -o ${OUTPUT_FILE} ${outfile_local_y2u} ${outfile_local_y2v}

mkdir -p ${OUTPUT_DIR}

# copy the file
scp ${OUTPUT_FILE} ${OUTPUT_DIR}/${OUTPUT_FILE}