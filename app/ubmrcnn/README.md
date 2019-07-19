# UB Mask-RCNN Ancestor Clustering Worker/Client

## To Run

These instructions assume you are in the container.
We will point out modifications for other environments.
We assume you've build all the dependencies.

First you need setup environment variables for the 
different dependencies by sourcing the following scripts.


      source [ubdl]/scripts/container_setenv.sh
      source [ubdl]/configure.sh
      cd ublarcvserver
      source configure.sh
      cd app/ubmrcnn
      source setenv.sh


with `[ubdl]` being the `ubdl` repository top-directory.
The first `container_setenv.sh` might be different for your system.
The `[ubdl]/script` folder has some examples for different machines.

Get the weights if you haven't yet.

To run a broker ...


To run a worker ...

To run a client ...


