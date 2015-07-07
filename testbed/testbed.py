# -*- coding: utf-8 -*-

import time
import string

import numpy
import theano
from theano import tensor as T

import dnn
from dnn.model import SdAIndividual
import utils

class TestBed(object):
    def __init__(self, window_size=100, n=2, w=10, h=10, d=1, hidden_layers_sizes=[100]):
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
        self.model = SdAIndividual(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)

    def supply(self, data):
        self.dataset.append(data)
        while self.window_size < len(self.dataset):
            self.dataset.pop(0)

    def pretrain(self, epochs=100, learning_rate=0.1):
        '''
        現在持っているデータセットで学習する
        :return:
        '''
        return self.model.pretrain(self.dataset, epochs=epochs, learning_rate=learning_rate)

    def finetune(self, epochs=100, learning_rate=0.1):
        '''
        現在持っているデータセットで学習する
        :return:
        '''
        return self.model.finetune(self.dataset, epochs=epochs, learning_rate=learning_rate)

    def predict(self, x=None):
        '''
        入力 x から y の値を予測する
        :param x: d-by-w-by-h 次元の ndarray のデータが r 個入った配列
        :return:
        '''
        if x is None:
            x = self.dataset[-self.n:]
        return self.model.predict(x)
