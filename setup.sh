#!/bin/bash

home=$PWD

# LIBTORCH
export LIBTORCH_LIBDIR="/home/jmills/.local/lib/python3.5/site-packages/torch/lib"
[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}/bin:"* ]] && \
    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"

# ROOT
source /home/twongj01/software/root6/python3build/bin/thisroot.sh

# LARCV
cd /home/jmills/workdir/ubdl
source configure.sh

cd $home
