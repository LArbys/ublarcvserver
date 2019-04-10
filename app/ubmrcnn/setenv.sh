#!/bin/bash

# location of this script
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# add model folder to python path
export UBMRCNN_MODEL_DIR=${UBLARCVSERVER_BASEDIR}/networks/ubmrcnn/models
export UBMRCNN_WEIGHT_DIR=${UBLARCVSERVER_BASEDIR}/networks/ubmrcnn/weights
export UBMRCNN_SERVERAPP_DIR=$where

# add to python path

# 1) model
[[ ":$PYTHONPATH:" != *":${UBMRCNN_MODEL_DIR}:"* ]] && \
    export PYTHONPATH="${UBMRCNN_MODEL_DIR}:${PYTHONPATH}"

# 2) server app
[[ ":$PYTHONPATH:" != *":${UBMRCNN_SERVERAPP_DIR}:"* ]] && \
    export PYTHONPATH="${UBMRCNN_SERVERAPP_DIR}:${PYTHONPATH}"

# PATH
[[ ":$PATH:" != *":${UBMRCNN_SERVERAPP_DIR}:"* ]] && \
    export PATH="${UBMRCNN_SERVERAPP_DIR}:${PATH}"
