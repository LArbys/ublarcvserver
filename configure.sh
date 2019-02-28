#!/bin/bash

export UBLARCVSERVER_BASEDIR=$PWD
export UBLARCVSERVER_BUILDDIR=${UBLARCVSERVER_BASEDIR}/build

export UBLARCVSERVER_INSTALL_DIR=${UBLARCVSERVER_BASEDIR}/release
export UBLARCVSERVER_LIBDIR=${UBLARCVSERVER_INSTALL_DIR}/lib
export UBLARCVSERVER_INCDIR=${UBLARCVSERVER_INSTALL_DIR}/include

# dependency majordomo library
export LIBMDP_LIBDIR=${UBLARCVSERVER_INSTALL_DIR}/libmdp/lib

# prepend ublarcvserver libdir to ld library path
[[ ":$LD_LIBRARY_PATH:" != *":${UBLARCVSERVER_LIBDIR}/bin:"* ]] && \
    export LD_LIBRARY_PATH="${UBLARCVSERVER_LIBDIR}:${LD_LIBRARY_PATH}"

# prepend libmdp libdir to ld lib path
[[ ":$LD_LIBRARY_PATH:" != *":${LIBMDP_LIBDIR}/bin:"* ]] && \
    export LD_LIBRARY_PATH="${LIBMDP_LIBDIR}:${LD_LIBRARY_PATH}"

# add to python path
[[ ":$PYTHONPATH:" != *":${UBLARCVSERVER_BASEDIR}/python:"* ]] && \
    export PYTHONPATH="${UBLARCVSERVER_BASEDIR}/python:${PYTHONPATH}"



