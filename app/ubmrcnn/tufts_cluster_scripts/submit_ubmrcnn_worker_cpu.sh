#!/bin/bash
#
#SBATCH --job-name=ubmrcnn_worker_cpu
#SBATCH --output=slurmout_ubmrcnn_cpu_worker.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=2
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

# WEIGHT DIR
WEIGHT_DIR=/cluster/tufts/wongjiradlab/larbys/dl_models/ubmrcnn_mcc8_v1

# WORK LOG
WORKER_LOG=${WORKDIR}/worker_cpu_ubmrcnn.log

# IP ADDRESSES OF BROKER
#BROKER=10.246.81.73 # PGPU03
BROKER=nudot.lns.mit.edu

PORT=6000

# GPU LIST
GPU_ASSIGNMENTS=${WORKDIR}/tufts_pgpu03_assignments.txt

COMMAND="source ${WORKDIR}/run_cpu_ubmrcnn_worker.sh tcp://${BROKER}:${PORT} ${UBMRCNN_DIR} ${UBDL_DIR} ${WEIGHT_DIR} ${GPU_ASSIGNMENTS} ${WORKER_LOG}"

module load singularity
singularity exec ${CONTAINER} bash -c "${COMMAND}"
