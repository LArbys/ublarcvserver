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

CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031118.img
SSS_BASEDIR=/usr/local/ubdl
WORKDIR=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/ubdlserver

# IP ADDRESSES OF BROKER
BROKER=10.246.81.73 # PGPU03
# BROKER=10.X.X.X # ALPHA001

PORT=6000

# GPU LIST
GPU_ASSIGNMENTS=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/serverssnet/tufts_pgpu03_assignments.txt
#GPU_ASSIGNMENTS=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/serverssnet/temptufts_pgpu03_assignments.txt
WORKEROFFSET=0

module load singularity
singularity exec --nv ${CONTAINER} bash -c "source ${WORKDIR}/run_ubmrcnn_worker.sh"
