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

## notes:

last part is to embed python routine responsible for making images
into the c++ framework.

want user to be able to write a module that looks like the following:

```
from larcv import larcv
from ublarcvserver import ublarcvserver
from ublarcvserver import ublarcvserver.util as util
import torch


model = load_model(...)

def process_net( img_list_bson_str ):
  """
  img_list_bson_str is a list of pystrings, with the contents being:
    ["imgtype",imgdata,"imgtype",imgdata,...]       
  """
  global model

  img_v = util.stringlist2img2(img_list_bson_str)
  img_np = [ larcv.as_ndarray(img_v.at(iimg)) for iimg in xrange(img_v.size()) ]

  ...

  out = model.forward(batch)

  ...

  # convert tensor to image2d to bson list

  return imgout_list_bson_str

```

the goal is to reduce the amount of non-network code the user has to write.

seems doable following something like this [example](https://docs.python.org/2/extending/embedding.html).
