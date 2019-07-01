#!/bin/bash
#
#SBATCH --job-name=splarflow_worker_pgpu03
#SBATCH --output=slurmout_splarflow_worker.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03
#SBATCH --array=14

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_051919.img

# UBDL_DIR: location of ubdl dependencies (e.g. larlite, larcv, ublarcvapp)
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl
#UBDL_DIR=/usr/local/ubdl # Container copy

# UBLARFLOW_DIR: location of UB Mask-RCNN scripts
UBLARFLOW_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/ublarcvserver/app/ublarflow
#UBLARFLOW_DIR=${UBDL_DIR}/ublarcvserver/app/ublarflow  # Container copy

# WORKING DIRECTORY
WORKDIR=${UBLARFLOW_DIR}/tufts_cluster_scripts

# WEIGHT DIR
WEIGHT_DIR=/cluster/tufts/wongjiradlab/larbys/dl_models/sparse_larflow/v1/

# WORK LOG
WORKER_LOGDIR=${WORKDIR}/

# IP ADDRESSES OF BROKER
#BROKER=10.246.81.73 # PGPU03
BROKER=nudot.lns.mit.edu

PORT=6000

# GPU LIST
GPU_ASSIGNMENTS=${WORKDIR}/tufts_pgpu03_assignments.txt

COMMAND="source ${WORKDIR}/run_sparse_larflow_worker.sh tcp://${BROKER}:${PORT} ${UBLARFLOW_DIR} ${UBDL_DIR} ${WEIGHT_DIR} ${GPU_ASSIGNMENTS} ${WORKER_LOGDIR}"

module load singularity
singularity exec --nv ${CONTAINER} bash -c "${COMMAND}"
