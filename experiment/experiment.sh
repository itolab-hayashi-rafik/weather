#!/bin/bash
#
# a simple script to run experiments
#

#THEANO_FLAGS='device=gpu0' nohup python -u moving_mnist.py > logs/moving_mnist.log 2>&1 &
THEANO_FLAGS='device=gpu0' nohup python -u weather_data_radar.py > logs/weather_data_radar.log 2>&1 &
THEANO_FLAGS='device=gpu2' nohup python -u weather_data_sat1.py > logs/weather_data_sat1.log 2>&1 &