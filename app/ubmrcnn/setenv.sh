#!/bin/bash

# location of this script
where="$( cd "$( dirname "${PWD}/${BASH_SOURCE[0]}" )" && pwd )"

# add model folder to python path
export UBMRCNN_SERVERAPP_DIR=$where

# add detectron folder to python path
export MASKRCNN_DIR=${UBLARCVSERVER_BASEDIR}/networks/mask-rcnn.pytorch

# add to python path

# 1) server app
[[ ":$PYTHONPATH:" != *":${UBMRCNN_SERVERAPP_DIR}:"* ]] && \
    export PYTHONPATH="${UBMRCNN_SERVERAPP_DIR}:${PYTHONPATH}"

# 2a) detectron
[[ ":$PYTHONPATH:" != *":${MASKRCNN_DIR}:"* ]] && \
    export PYTHONPATH="${MASKRCNN_DIR}:${PYTHONPATH}"

# 2b) detectron lib
[[ ":$PYTHONPATH:" != *":${MASKRCNN_DIR}/lib:"* ]] && \
    export PYTHONPATH="${MASKRCNN_DIR}/lib:${PYTHONPATH}"

# 3) PATH
[[ ":$PATH:" != *":${UBMRCNN_SERVERAPP_DIR}:"* ]] && \
    export PATH="${UBMRCNN_SERVERAPP_DIR}:${PATH}"
