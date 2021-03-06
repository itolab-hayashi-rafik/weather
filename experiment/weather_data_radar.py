# -*- coding: utf-8 -*-
__author__ = 'masayuki'

import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import datetime

from testbed import experiment as E

if __name__ == '__main__':
    train_dataset='../data/weather_data/out_radar/dataset-train.npz'
    valid_dataset='../data/weather_data/out_radar/dataset-valid.npz'
    test_dataset='../data/weather_data/out_radar/dataset-test.npz'
    patchsize = (8,8)

    n_feature_maps = patchsize[0]*patchsize[1]
    filter_shapes = [(32,n_feature_maps,3,3),(32,32,3,3)]

    now = datetime.datetime.today()
    saveto = "out/{0}-states-{1}.npz".format(__file__,now.strftime('%Y%m%d%I%M%S'))
    print('Save file: {0}'.format(saveto))

    print('begin experiment')
    E.exp_moving_mnist(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        patch_size=patchsize,
        filter_shapes=filter_shapes,
        saveto=saveto)
    print('finish experiment')