#!/bin/bash

home=$PWD

# LIBTORCH
export LIBTORCH_LIBDIR="/usr/local/torchlib/lib"
[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}/bin:"* ]] && \
    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"

# ROOT
source ~/setup_root6.sh

# LARCV
cd ~/working/larbys/ubdl
source configure.sh

cd $home


