#!/bin/bash

# location of this script
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# add model folder to python path
export UBLARFLOW_SERVERAPP_DIR=$where

# add to python path

# 1) model
#[[ ":$PYTHONPATH:" != *":${UBLARFLOW_MODEL_DIR}:"* ]] && \
#    export PYTHONPATH="${UBLARFLOW_MODEL_DIR}:${PYTHONPATH}"

# 2) server app
#[[ ":$PYTHONPATH:" != *":${UBLARFLOW_SERVERAPP_DIR}:"* ]] && \
#    export PYTHONPATH="${UBLARFLOW_SERVERAPP_DIR}:${PYTHONPATH}"

# PATH
[[ ":$PATH:" != *":${UBLARFLOW_SERVERAPP_DIR}:"* ]] && \
    export PATH="${UBLARFLOW_SERVERAPP_DIR}:${PATH}"
