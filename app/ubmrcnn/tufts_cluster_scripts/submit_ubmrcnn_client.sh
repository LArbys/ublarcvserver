#!/bin/bash
#
#SBATCH --job-name=ubmrcnn_client_cpu
#SBATCH --output=slurmout_ubmrcnn_cpu_client.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition batch
#SBATCH --array=0

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_py3_deponly_070119.img


# UBDL_DIR: location of ubdl dependencies (e.g. larlite, larcv, ublarcvapp)
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/py3_ubdl
#UBDL_DIR=/usr/local/ubdl # Container copy

# UBMRCNN_DIR: location of UB Mask-RCNN scripts
UBMRCNN_DIR=/cluster/tufts/wongjiradlab/twongj01/py3_ubdl/ublarcvserver/app/ubmrcnn
#UBMRCNN_DIR=${UBDL_DIR}/ublarcvserver/app/ubmrcnn  # Container copy

# WORKING DIRECTORY
WORKDIR=${UBMRCNN_DIR}/tufts_cluster_scripts

# WORK LOG
WORKER_LOG=${WORKDIR}/client_ubmrcnn.log

# IP ADDRESSES OF BROKER
#BROKER=10.246.81.73 # PGPU03
BROKER=nudot.lns.mit.edu

PORT=6000

SUPERA_INPUTPATH=/cluster/kappa/90-days-archive/wongjiradlab/larbys/data/db/mcc9_v13_nueintrinsics_overlay_run1/stage1/049/52/000/67//supera-Run004952-SubRun000067.root
OUTPUT_FILE=test.root

COMMAND="source ${WORKDIR}/run_ubmrcnn_client.sh tcp://${BROKER}:${PORT} ${WORKDIR} ${UBMRCNN_DIR} ${UBDL_DIR} ${SUPERA_INPUTPATH} ${OUTPUT_FILE}"

module load singularity
singularity exec ${CONTAINER} bash -c "${COMMAND}"
