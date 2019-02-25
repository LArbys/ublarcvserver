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


