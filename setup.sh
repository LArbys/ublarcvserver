#!/bin/bash

export LIBTORCH_LIBDIR="/usr/local/torchlib/lib"


[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}/bin:"* ]] && \
    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"

