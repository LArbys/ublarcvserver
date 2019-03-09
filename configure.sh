#!/bin/bash

export UBLARCVSERVER_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# add to python path
[[ ":$PYTHONPATH:" != *":${UBLARCVSERVER_BASEDIR}/python:"* ]] && \
    export PYTHONPATH="${UBLARCVSERVER_BASEDIR}/python:${PYTHONPATH}"



