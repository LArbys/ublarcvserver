#!/bin/bash

#set username
username=twongj01

# preliminary sparse weights: June 2019
rsync -av --progress $username@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/larbys/dl_models/sparse_infill/prelim_june2019/* prelim_june2019/
