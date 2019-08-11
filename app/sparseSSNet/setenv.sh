#!/bin/bash

# location of this script
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# add model folder to python path
export SPARSE_URESNET_SERVERAPP_DIR=$where

# add detectron folder to python path
export SPARSE_URESNET_DIR=${UBLARCVSERVER_BASEDIR}/networks/sparse_ssnet

# add to python path

# 1) server app
[[ ":$PYTHONPATH:" != *":${SPARSE_URESNET_SERVERAPP_DIR}:"* ]] && \
    export PYTHONPATH="${SPARSE_URESNET_SERVERAPP_DIR}:${PYTHONPATH}"

# 2) sparse uresnet
[[ ":$PYTHONPATH:" != *":${SPARSE_URESNET_DIR}:"* ]] && \
    export PYTHONPATH="${SPARSE_URESNET_DIR}:${PYTHONPATH}"

# 3) PATH
[[ ":$PATH:" != *":${SPARSE_URESNET_SERVERAPP_DIR}:"* ]] && \
    export PATH="${SPARSE_URESNET_SERVERAPP_DIR}:${PATH}"
