# ublarcvserver

The main use case of this repository is to process images from LArTPC detectors
through deep neural networks using remote GPU workers.

We build this using the [MajorDomo project](https://github.com/zeromq/majordomo) which is a
"service-oriented broker" that allows workers providing different services.
This is useful when we want to process several types of neural networks.
For MicroBooNE this includes:

* semantic segmentation
* larflow
* infill
* flash-level clustering
* individual particle level clustering of neutrino candidate cluster

# To Do List

## for minimal working setup

* broker class [done]
* worker class base [done]
* worker class dummy [done]
* client class base [done]
* worker class larcv image passing [done]
* worker class larcv image passing base [done]
* worker class larcv image + network processing

## test setup

* python binding test of simple dummy client and broker [done]
* python binding test of larcv image2d passing protocol [done]

## real applications

* caffe (from python) ssnet service (translation of existing ssnetserver)
* pytorch ssnet service using torch script
* pytorch ssnet service for sparse convnet

## additional features
