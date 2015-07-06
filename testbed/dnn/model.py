# -*- coding: utf-8 -*-

import numpy

from network.SdA import SdA


class SdAIndividual(object):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, hidden_layers_sizes=[10]):
        self.n = n
        self.d = d
        self.w = w
        self.h = h
        self.n_hidden_layers = len(hidden_layers_sizes)

        self.dnns = \
        [
            [
                SdA(
                    numpy_rng=numpy_rng,
                    n_ins=n,
                    hidden_layers_sizes=hidden_layers_sizes,
                    n_outs=d
                ) for i in xrange(w)
            ] for j in xrange(h)
        ]

        self.pretrain_fns = [ [dnn.pretraining_functions() for dnn in dnns] for dnns in self.dnns ]
        self.finetune_fns = [ [dnn.build_finetune_function() for dnn in dnns] for dnns in self.dnns ]
        self.predict_fns = [ [dnn.build_prediction_function() for dnn in dnns] for dnns in self.dnns ]

    def _make_input(self, data, i, j):
        '''
        (i,j) の SdA に対する入力ベクトルを data から作る
        :param data: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :param i:
        :param j:
        :return:
        '''
        return numpy.append([], [chunk[:,j,i] for chunk in data])

    def _make_output(self, data, i, j):
        return data[:,j,i]

    def pretrain(self, dataset, epochs=100, learning_rate=0.1):
        '''
        pretrain the model using the dataset
        :param dataset:
        :param epochs:
        :param learning_rate:
        :return:
        '''
        # TODO; epochs 使う
        idx = range(self.n,len(dataset))
        numpy.random.shuffle(idx)

        avg_cost = 0.0

        for l in xrange(self.n_hidden_layers):
            for t in idx:
                xdata = dataset[(t-self.n):t-1]
                for j in xrange(self.h):
                    for i in xrange(self.w):
                        x = self._make_input(xdata, i, j)
                        f = self.pretrain_fns[j][i][l]
                        cost = f(x, lr=learning_rate)
                        avg_cost += cost

        avg_cost /= len(idx) * (self.w*self.h)

        return avg_cost

    def finetune(self, dataset, epochs=100, learning_rate=0.1):
        '''
        finetune the model using the dataset
        :param dataset: an array of ndarray of (d-by-h-by-w) dimention, whose size is bigger than n
        :return:
        '''
        # TODO; epochs 使う
        idx = range(self.n,len(dataset))
        numpy.random.shuffle(idx)

        avg_cost = 0.0

        for t in idx:
            xdata = dataset[(t-self.n):t-1]
            ydata = dataset[t]
            for j in xrange(self.h):
                for i in xrange(self.w):
                    x = self._make_input(xdata, i, j)
                    y = self._make_output(ydata, i, j)
                    f = self.finetune_fns[j][i]
                    cost = f(x, y, lr=learning_rate)
                    avg_cost += cost

        avg_cost /= len(idx) * (self.w*self.h)

        return avg_cost

    def predict(self, x):
        '''
        predict the next value
        :param x: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        y = \
        [
            [
                self.predict_fns(self._make_input(x, i, j))
                for i in xrange(self.w)
            ] for j in xrange(self.h)
        ]

        return y