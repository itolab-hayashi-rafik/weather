#!/bin/bash
#
# a simple script to run experiments
#

#THEANO_FLAGS='device=gpu0' nohup python -u experiment.py 1 > logs/1.log 2>&1 &
#THEANO_FLAGS='device=gpu2' nohup python -u experiment.py 2 > logs/2.log 2>&1 &
THEANO_FLAGS='device=gpu0' nohup python -u experiment.py 3 > logs/3.log 2>&1 &
#THEANO_FLAGS='device=gpu0' nohup python -u experiment.py 4 > logs/4.log 2>&1 &
#THEANO_FLAGS='device=gpu0' nohup python -u experiment.py 5 > logs/5.log 2>&1 &
#THEANO_FLAGS='device=gpu0' nohup python -u experiment.py 6 > logs/6.log 2>&1 &