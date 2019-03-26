#!/bin/bash

# add to python path
where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export UBLARFLOW_NET_BASEDIR=$where
[[ ":$PYTHONPATH:" != *":${UBLARFLOW_NET_BASEDIR}/models:"* ]] && \
    export PYTHONPATH="${UBLARFLOW_NET_BASEDIR}/models:${PYTHONPATH}"
