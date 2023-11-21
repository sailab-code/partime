# PARTIME: Scalable and Parallel Processing Over Time with Deep Neural Networks

This directory contains the code of the PARTIME library presented in the manuscript ["PARTIME: Scalable and Parallel Processing Over
Time with Deep Neural Networks"](https://ieeexplore.ieee.org/document/10068971).

[Technical report](https://arxiv.org/abs/2210.09147)

## Requirements
The main requirements for this library are
 - PyTorch >= 11 with CUDA >= 11.3
 - wandb (for experimentD.py logging of accuracy over time)
 - torchvision
 - pandas

## Content

The directory contains two subdirectories and 5 python files that help replicate the experiments described in the paper.

### Directories
 - extras: contains code used to generate sequential versions of the ResNet
 - partime: contains the PARTIME library

### Files

 - common.py: contains common code used in other scripts, mainly pipeline execution and time measurement
 - experimentA.py: contains the code to replicate V.A - Launch with `python ./experimentA.py`
 - experimentB.py: contains the code to replicate V.B - Launch with `python ./experimentB.py`
 - experimentC.py: contains the code to replicate V.C - Launch with `python ./experimentC.py`
 - experimentD.py: contains the code to replicate V.D - Launch with `python ./experimentD.py --lr=<lr> --batch_size=<batch_size> --stages=<n_stages> --max_epochs=<n_epochs>`. The script contains the code to log data with wandb. If wandb is not to be used, comment the import and set `DEBUG=True` at line 27

Acknowledgement
---------------

This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).
