#!/bin/bash

export UBLARCVSERVER_BASEDIR=$PWD

# add to python path
[[ ":$PYTHONPATH:" != *":${UBLARCVSERVER_BASEDIR}/python:"* ]] && \
    export PYTHONPATH="${UBLARCVSERVER_BASEDIR}/python:${PYTHONPATH}"



