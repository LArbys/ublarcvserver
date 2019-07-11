#!/bin/bash

# configure all the environment variables for this repo.
# assumed we live inside the ubdl repository

# get location where script is called
__ublarflow_configureall_start_dir__=`pwd`
echo "START DIR: ${start_dir}"

# location of this script
where="$( cd "$( dirname "${PWD}/${BASH_SOURCE[0]}" )" && pwd )"
echo "LOCATION OF ${BASH_SOURCE}: ${where}"

# setup dependencies
echo "SETUP UBDL"
cd ${where}/../../../ || ( cd $__ublarflow_configureall_start_dir__= && exit )
source setenv.sh || ( cd $__ublarflow_configureall_start_dir__= && exit )
export LARCV_OPENCV=0
source configure.sh || ( cd $__ublarflow_configureall_start_dir__= && exit )

echo "SETUP LARFLOW"
cd larflow
source configure.sh
cd ../

echo "SETUP UBLARCVSERVER"
cd ublarcvserver || ( cd $__ublarflow_configureall_start_dir__= && exit )
source configure.sh || ( cd $__ublarflow_configureall_start_dir__= && exit )
cd ${__ublarflow_configureall_start_dir__=}

# setup the script dir
echo "SETUP UB LARFLOW"
cd ${__ublarflow_configureall_start_dir__}
source setenv.sh || ( cd $__ublarflow_configureall_start_dir__= && exit )
