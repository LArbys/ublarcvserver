# ublarcvserver

The main use case of this repository is to process images from LArTPC detectors
through deep neural networks using remote GPU workers.

We build this using the [MajorDomo project](https://github.com/zeromq/majordomo) which is a
"service-oriented broker" that allows workers providing different services.
This is useful when we want to process several types of neural networks.

This solution uses, [majortomo](https://github.com/shoppimon/majortomo),
a purely python implementation of the MajorDomo protocol.

Majortomo is (C) Copyright 2018 Shoppimon LTD and
distributed under the terms of the Apache 2.0 License (see LICENSE).

## Applications to be supported

For MicroBooNE this code is to handle:

* ubSSNet (in `app/ubssnet`): track/shower semantic segmentation
  trained using caffe+MCC8 particle gun data
* larflow (to do)
* infill (to do)
* flash-level clustering (to do)
* individual particle level clustering of neutrino candidate cluster (to do)

# Instructions

To do

# To Do List

* instructions
* basic/generic torch-script worker and larcv client
* better compression (mostly about how one does the image2d -> bson conversion)
* submodules to network repos
* container support and tufts cluster scripts



# Misc notes:

## goals for easy-to-use, generic application

for generic application, want user to be able to write a module that looks like the following:

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
