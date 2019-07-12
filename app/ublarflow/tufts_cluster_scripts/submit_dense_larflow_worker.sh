#!/bin/bash
#
#SBATCH --job-name=dense_larflow_worker_batch2
#SBATCH --output=slurmout_dense_larflow_worker_batch2.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu02
#SBATCH --array=1-3

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_070119.img

# UBDL_DIR: location of ubdl dependencies (e.g. larlite, larcv, ublarcvapp)
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl
#UBDL_DIR=/usr/local/ubdl # Container copy

# UBLARFLOW_DIR: location of UB Mask-RCNN scripts
UBLARFLOW_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/ublarcvserver/app/ublarflow
#UBLARFLOW_DIR=${UBDL_DIR}/ublarcvserver/app/ublarflow  # Container copy

# WORKING DIRECTORY
WORKDIR=${UBLARFLOW_DIR}/tufts_cluster_scripts

# WEIGHT DIR
WEIGHT_DIR=/cluster/tufts/wongjiradlab/larbys/dl_models/dense_larflow/

# WORK LOG
WORKER_LOGDIR=${WORKDIR}/

# IP ADDRESSES OF BROKER
#BROKER=10.246.81.73 # PGPU03
BROKER=nudot.lns.mit.edu

PORT=6000

# GPU LIST
GPU_ASSIGNMENTS=${WORKDIR}/tufts_pgpu03_assignments.txt

COMMAND="source ${WORKDIR}/run_dense_larflow_worker.sh tcp://${BROKER}:${PORT} ${UBLARFLOW_DIR} ${UBDL_DIR} ${WEIGHT_DIR} ${GPU_ASSIGNMENTS} ${WORKER_LOGDIR}"

module load singularity
singularity exec --nv ${CONTAINER} bash -c "${COMMAND}"
