#!/bin/bash

#SBATCH --job-name=prep_pgpu03
#SBATCH --output=prep_pgpu03.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=10:00
#SBATCH --cpus-per-task=1
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03

# This script copies
# 1) network description: dllee_ssnet2018.prototxt
# 2) network weights: 

SSS_BASEDIR=/cluster/kappa/90-days-archive/wongjiradlab/larbys/ssnetserver
SSS_MODELDIR=/cluster/kappa/90-days-archive/wongjiradlab/larbys/ssnet_models/v1/

rsync -av --progress ${SSS_BASEDIR}/dllee_ssnet2018.prototxt /tmp/
rsync -av --progress ${SSS_MODELDIR}/*.caffemodel /tmp/

ls -lh /tmp/*.prototxt
ls -lh /tmp/*.caffemodel