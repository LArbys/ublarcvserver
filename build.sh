#!/bin/bash

# check if envirnoment setup
if [ -z "$UBLARCVSERVER_BASEDIR" ]; then
    echo "UBLARCVSERVER_BASEDIR is not set. Run configure.sh"
    exit 1
fi

# build major domo
if [ ! -f ${UBLARCVSERVER_BASEDIR}/release/libmdp/lib/libmajordomo.so ]; then
  cd majordomo
  ./autogen.sh
  ./configure --prefix=${UBLARCVSERVER_BASEDIR}/release
  make && make install
fi

mkdir build
cd build
cmake ../
make
