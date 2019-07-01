#!/bin/bash
#
#SBATCH --job-name=sparseinfill_worker_pgpu03
#SBATCH --output=ubdl_workers_sparseinfill.log
#SBATCH --mem-per-cpu=12000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu02
#SBATCH --array=[2-3]

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_051919.img

# UBDL_DIR: location of dependencies (e.g. larcv, larlite, ublarcvapp)
#UBDL_DIR=/cluster/tufts/wongjiradlab/kmason03/testdir/ubdl
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl
#UBDL_DIR=/usr/local/ubdl # In container

# INFILL_DIR: location of ublarcvserver Infill app directory
INFILL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/ublarcvserver/app/UBInfill

# WEIGHT_DIR: location of weights
WEIGHT_DIR=/cluster/tufts/wongjiradlab/larbys/dl_models/sparse_infill/prelim_june2019/

# WORKDIR: place for scripts
WORKDIR=${INFILL_DIR}/sparseinfill

# GPU_LIST
#GPU_LIST=/cluster/tufts/wongjiradlab/kmason03/testdir/ubdl/ublarcvserver/app/UBInfill/sparseinfill/tufts_pgpu03_assignments.txt
GPU_LIST=${WORKDIR}/tufts_pgpu03_assignments.txt

# LOG_FILE
#LOG_DIR=/tmp/
LOG_DIR=${WORKDIR}/


# IP ADDRESSES OF BROKER
#BROKER=10.246.81.73 # PGPU03
#BROKER=10.246.81.72 #PGPU02
# BROKER=10.X.X.X # ALPHA001
BROKER=nudot.lns.mit.edu

PORT=6000

COMMAND="source ${WORKDIR}/run_sparseinfill_worker.sh tcp://${BROKER}:${PORT} ${UBDL_DIR} ${INFILL_DIR} ${WEIGHT_DIR} ${GPU_LIST} ${LOG_DIR}"

module load singularity
singularity exec --nv ${CONTAINER} bash -c "${COMMAND}"
