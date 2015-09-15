# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import time
import string

import numpy
import theano
from theano import tensor as T

import dnn
from generator import SinGenerator, RadarGenerator
import utils

class TestBed(object):
    def __init__(self, window_size=100, n=2, w=10, h=10, d=1, hidden_layers_sizes=[3]):
        '''
        初期化する
        :param window_size:
        :param n: DNN に入力する過去のデータの個数
        :param w: 各データの横幅
        :param h: 各データの高さ
        :param d: 各データのチャンネル数
        :param hidden_layers_sizes: 中間層のユニット数
        :return:
        '''
        self.window_size = window_size
        self.n = n
        self.w = w
        self.h = h
        self.d = d
        self.dataset = [ numpy.zeros((d,h,w), dtype=theano.config.floatX) for i in xrange(window_size) ]

        numpy_rng = numpy.random.RandomState(89677)
        # for each value n in hidden_layers_sizes, assume it as a filter of (1,d,n,n), which means it has one n*n sized filter
        filter_shapes = [(1,d,k,k) for k in hidden_layers_sizes]

        # self.model = dnn.SdAIndividual(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)
        # self.model = dnn.SdAFullyConnected(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)
        # self.model = dnn.LSTMFullyConnected(numpy_rng, n=n, d=d, w=w, h=h, hidden_layers_sizes=hidden_layers_sizes)
        # self.model = dnn.ConvLSTMFullyConnected(numpy_rng, n=n, d=d, w=w, h=h, filter_shapes=filter_shapes)
        # self.model = dnn.EncoderDecoderLSTM(numpy_rng, n=n, d=d, w=w, h=h, hidden_layers_sizes=hidden_layers_sizes)
        self.model = dnn.EncoderDecoderConvLSTM(numpy_rng, n=n, d=d, w=w, h=h, filter_shapes=filter_shapes)

    def supply(self, data):
        self.dataset.append(data)
        while self.window_size < len(self.dataset):
            self.dataset.pop(0)

    def pretrain(self, epochs=15, learning_rate=0.1, batch_size=1):
        '''
        現在持っているデータセットで学習する
        :return:
        '''
        return self.model.pretrain(
            numpy.asarray(self.dataset, dtype=theano.config.floatX),
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    def finetune(self, epochs=100, learning_rate=0.1, batch_size=1):
        '''
        現在持っているデータセットで学習する
        :return:
        '''
        idx = range(self.window_size-self.n)
        numpy.random.shuffle(idx)
        cut = int(0.8*len(idx))
        train_idx = idx[:cut]
        valid_idx = idx[cut:]
        return self.model.finetune(
            numpy.asarray(self.dataset, dtype=theano.config.floatX),
            train_idx=train_idx,
            valid_idx=valid_idx,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )

    def predict(self):
        '''
        入力 x から y の値を予測する
        :param x: d-by-w-by-h 次元の ndarray のデータが n 個入った配列
        :return:
        '''
        return self.model.predict(
            numpy.asarray(self.dataset, dtype=theano.config.floatX)
        )


if __name__ == '__main__':
    bed = TestBed()
    gen = SinGenerator(w=bed.w, h=bed.h, d=bed.d)
    # gen = RadarGenerator("../data/radar", w=bed.w, h=bed.h)

    # fill the window with data
    for i in xrange(bed.window_size):
        y = gen.next()
        bed.supply(y)

    for i,y in enumerate(gen):
        # predict
        y_pred = bed.predict()
        print("{}: y={}, y_pred={}".format(i, y, y_pred))

        bed.supply(y)

        # if i % pretrain_step == 0 and 0 < self.pretrain_epochs:
        #     # pretrain
        #     avg_cost = self.bed.pretrain(self.pretrain_epochs, learning_rate=self.pretrain_lr, batch_size=self.pretrain_batch_size)
        #     print("   pretrain cost: {}".format(avg_cost))
        #     pass

        # finetune
        avg_cost = bed.finetune(batch_size=7)
        print(" finetune {}, train cost: {}".format(i,avg_cost))

        time.sleep(1)
