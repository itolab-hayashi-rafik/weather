# -*- coding: utf-8 -*-
__author__ = 'masayuki'

import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import datetime

import numpy
import theano

from models.convlstm_encoder_decoder import ConvLSTMEncoderDecoder
from testbed.utils import ndarray


@profile
def check_memory(t_in=10, d=1, w=64, h=64, t_out=10, filter_shapes=[(1,1,3,3)]):
    # build model
    print('building model...'),
    numpy_rng = numpy.random.RandomState(89677)
    model = ConvLSTMEncoderDecoder(numpy_rng, t_in=t_in, d=d, w=w, h=h, t_out=t_out, filter_shapes=filter_shapes)
    f_grad_shared, f_update = model.build_finetune_function()
    f_predict = model.build_prediction_function()
    print('done')


if __name__ == '__main__':
    argv = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argv) # 引数の個数

    if argc <= 1:
        print("Usage: $ python {0} [1|2|3|4|5]".format(argv[0]))
        quit()

    exp = int(argv[1])
    if exp == 1:
        filter_shapes = [(256,1,5,5)]
    elif exp == 2:
        filter_shapes = [(128,1,5,5),(128,128,5,5)]
    elif exp == 3:
        filter_shapes = [(128,1,5,5),(64,128,5,5),(64,64,5,5)]
    elif exp == 4:
        filter_shapes = [(128,1,9,9),(128,128,9,9)]
    elif exp == 5:
        filter_shapes = [(128,1,9,9),(64,128,9,9),(64,64,9,9)]
    elif exp == 6:
        filter_shapes = [(2,1,5,5),(2,2,9,9)]
    else:
        raise NotImplementedError

    print('begin experiment')
    check_memory(filter_shapes=filter_shapes)
    print('finish experiment')