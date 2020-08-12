
**********************
Profile of Performance
**********************

This part of doc describes the performance in each cases.

CPU
===

Cuda
====

Formal
======

``conv2d`` is the most time-consuming operator. If we use the helping class ``Indices`` to tranverse over the output, ``test_model`` will be so slow that yolo_tfm cannot be finished until 15 minutes; if we don't use the helping class, however, test cases in ``test_op`` will be slower probabily because of the scale is small.


OpenCL
======
