#!/bin/bash

# location of this script
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# add model folder to python path
export UBMRCNN_MODEL_DIR=${UBLARCVSERVER_BASEDIR}/networks/ubmrcnn/models
export UBMRCNN_WEIGHT_DIR=${UBLARCVSERVER_BASEDIR}/networks/ubmrcnn/weights
export UBMRCNN_SERVERAPP_DIR=$where

# add detectron folder to python path
export MASKRCNN_DIR=${UBLARCVSERVER_BASEDIR}/networks/mask-rcnn.pytorch

# add to python path

# 1) model
[[ ":$PYTHONPATH:" != *":${UBMRCNN_MODEL_DIR}:"* ]] && \
    export PYTHONPATH="${UBMRCNN_MODEL_DIR}:${PYTHONPATH}"

# 2) server app
[[ ":$PYTHONPATH:" != *":${UBMRCNN_SERVERAPP_DIR}:"* ]] && \
    export PYTHONPATH="${UBMRCNN_SERVERAPP_DIR}:${PYTHONPATH}"

# 3a) detectron
[[ ":$PYTHONPATH:" != *":${MASKRCNN_DIR}:"* ]] && \
    export PYTHONPATH="${MASKRCNN_DIR}:${PYTHONPATH}"

# 3b) detectron lib
[[ ":$PYTHONPATH:" != *":${MASKRCNN_DIR}/lib:"* ]] && \
    export PYTHONPATH="${MASKRCNN_DIR}/lib:${PYTHONPATH}"


# PATH
[[ ":$PATH:" != *":${UBMRCNN_SERVERAPP_DIR}:"* ]] && \
    export PATH="${UBMRCNN_SERVERAPP_DIR}:${PATH}"
