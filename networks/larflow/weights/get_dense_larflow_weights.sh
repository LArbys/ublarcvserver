#!/bin/bash

user=$1

rsync -av --progress ${user}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev_filtered/devfiltered_larflow_*_832x512_32inplanes.tar .
