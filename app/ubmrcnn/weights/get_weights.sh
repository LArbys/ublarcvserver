#!/bin/bash

startdir=$PWD

# go to location of script
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# change me
username=twongj01
rsync -av --progress ${username}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/larbys/dl_models/ubmrcnn_mcc8_v1/mcc8_mrcnn_plane*.pth ${where}/ubmrcnn_mcc8_v1/ || cd $startdir

# go back to start dir
cd $startdir
