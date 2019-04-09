#!/bin/bash
#
#SBATCH --job-name=ubdl_workers_pgpu03
#SBATCH --output=ubdl_workers_pgpu03.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03
#SBATCH --array=0

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_python3_040619.img
SSS_BASEDIR=/cluster/tufts/wongjiradlab/jmills09/ubdl
WORKDIR=/cluster/tufts/wongjiradlab/jmills09/ubdl/ublarcvserver

# IP ADDRESSES OF BROKER
BROKER=10.246.81.72 # PGPU02
# BROKER=10.X.X.X # ALPHA001

PORT=6000

# GPU LIST
GPU_ASSIGNMENTS=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/serverssnet/tufts_pgpu03_assignments.txt
#GPU_ASSIGNMENTS=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/serverssnet/temptufts_pgpu03_assignments.txt
WORKEROFFSET=0

module load singularity
singularity exec --nv ${CONTAINER} bash -c "source ${WORKDIR}/run_ubmrcnn_worker.sh"
