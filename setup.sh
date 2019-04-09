#!/bin/bash

home_dir=$PWD

# LIBTORCH
export LIBTORCH_LIBDIR="/usr/local/lib/python3.5/dist-packages/torch/lib"
[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}/bin:"* ]] && \
    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"

# ROOT
source source /usr/local/root6-python3/bin/thisroot.sh
# LARCV
cd /cluster/tufts/wongjiradlab/jmills09/ubdl
source configure.sh

cd $home_dir
