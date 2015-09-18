# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import time
import string
import math

import numpy
import theano
from theano import tensor as T

import dnn
from generator import ConstantGenerator, SinGenerator, RadarGenerator
import utils

class TestBed(object):
    def __init__(self, window_size=10, t_in=5, w=10, h=10, d=1, t_out=1, hidden_layers_sizes=[3]):
        '''
        初期化する
        :param window_size:
        :param t_in: DNN に入力する過去のデータの個数
        :param w: 各データの横幅
        :param h: 各データの高さ
        :param d: 各データのチャンネル数
        :param t_out: DNN から出力する未来のデータの個数
        :param hidden_layers_sizes: 中間層のユニット数
        :return:
        '''
        self.window_size = window_size
        self.t_in = t_in
        self.w = w
        self.h = h
        self.d = d
        self.t_out = t_out
        self.dataset = [ numpy.zeros((d,h,w), dtype=theano.config.floatX) for i in xrange(window_size) ]

        numpy_rng = numpy.random.RandomState(89677)
        # for each value n in hidden_layers_sizes, assume it as a filter of (1,d,sqrt(n),sqrt(n)), which means it has one sqrt(n)*sqrt(n) sized filter
        filter_shapes = [(1,d,k,k) for k in hidden_layers_sizes]

        # self.model = dnn.SdAIndividual(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)
        # self.model = dnn.SdAFullyConnected(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)

        # StackedLSTM を使う場合は hidden_layers_sizes が [...] + [n_ins] でないといけない.
        # self.model = dnn.StackedLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, hidden_layers_sizes=hidden_layers_sizes)

        # StackedConvLSTM では中間層の大きさは入力層と同じ(固定). ただしパラメータ数(フィルタの数, 大きさ)は自由に変えられる.
        # self.model = dnn.StackedConvLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, filter_shapes=filter_shapes)

        # EncoderDecoderLSTM を使う場合は hidden_layers_sizes が [n_ins] + [...] + [n_ins] でないといけない.
        # self.model = dnn.EncoderDecoderLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, t_out=t_out, hidden_layers_sizes=hidden_layers_sizes)

        # EncoderDecoderConvLSTM では中間層の大きさは入力層と同じ(固定). ただしパラメータ数(フィルタの数, 大きさ)は自由に変えられる.
        self.model = dnn.EncoderDecoderConvLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, t_out=t_out, filter_shapes=filter_shapes)

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
        idx = range(self.window_size-self.t_in-self.t_out+1)
        numpy.random.shuffle(idx)
        cut = int(math.ceil(0.8*len(idx)))
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
        現在のデータセットから将来のデータを予測する
        :return:
        '''
        return self.model.predict(
            numpy.asarray(self.dataset, dtype=theano.config.floatX)
        )

    def save_params(self):
        params = self.model.params
        # TODO


if __name__ == '__main__':
    bed = TestBed()
    # gen = ConstantGenerator(w=bed.w, h=bed.h, d=bed.d)
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
        avg_cost = bed.finetune()
        print(" finetune {}, train cost: {}".format(i,avg_cost))

        time.sleep(1)
