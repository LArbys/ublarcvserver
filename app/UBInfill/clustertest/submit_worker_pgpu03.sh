#!/bin/bash
#
#SBATCH --job-name=ubdl_workers_pgpu03_0
#SBATCH --output=ubdl_workers_pgpu03_6.log
#SBATCH --mem-per-cpu=12000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu02
#SBATCH --array=2

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_032919.img
SSS_BASEDIR=/usr/local/ubdl
WORKDIR=/cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl/ublarcvserver/app/UBInfill/clustertest

# IP ADDRESSES OF BROKER
#BROKER=10.246.81.73 # PGPU03
BROKER=10.246.81.72 #PGPU02
# BROKER=10.X.X.X # ALPHA001

PORT=6000

# GPU LIST
GPU_ASSIGNMENTS=/cluster/tufts/wongjiradlab/kmason03/uboonecode/ubdl/ublarcvserver/app/UBInfill/clustertest/tufts_pgpu03_assignments.txt
#GPU_ASSIGNMENTS=/cluster/kappa/wongjiradlab/larbys/pubs/dlleepubs/serverssnet/temptufts_pgpu03_assignments.txt
WORKEROFFSET=0

module load singularity
singularity exec --nv ${CONTAINER} bash -c "source ${WORKDIR}/run_infill_worker.sh"
