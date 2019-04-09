#!/bin/bash
#
#SBATCH --job-name=ubdl_server
#SBATCH --output=ubdl_server.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=3
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_python3_040619.img
SERVER_BASEDIR=/cluster/tufts/wongjiradlab/jmills09/ubdl
WORKDIR=/cluster/tufts/wongjiradlab/jmills09/ubdl/ublarcvserver

module load singularity
singularity exec ${CONTAINER} bash -c "cd ${SERVER_BASEDIR} && source setenv.sh && source configure.sh \
&& cd scripts && source setenv_ublarcvserver.sh && \
./start_ublarcvserver_broker.py -l ${WORKDIR}/ubdl_server.log 6000"
