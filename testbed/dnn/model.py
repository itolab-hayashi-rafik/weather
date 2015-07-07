# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

from network.SdA import SdA

class SdAIndividual(object):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, hidden_layers_sizes=[10]):
        self.n = n
        self.d = d
        self.w = w
        self.h = h
        self.n_hidden_layers = len(hidden_layers_sizes)

        print('SdAIndividual: building the model...'),
        self.dnns = \
        [
            [
                SdA(
                    numpy_rng=numpy_rng,
                    n_ins=n*d,
                    hidden_layers_sizes=hidden_layers_sizes,
                    n_outs=d
                ) for i in xrange(w)
            ] for j in xrange(h)
        ]
        self.sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=n,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=d
        )
        print('done')

        print('SdAIndividual: building pretrain function...'),
        self.pretrain_fns = self.sda.pretraining_functions()
        print('done')

        print('SdAIndividual: building finetune function...'),
        self.finetune_fn = self.sda.build_finetune_function()
        print('done')

        print('SdAIndividual: building predict function...'),
        self.predict_fn = self.sda.build_prediction_function()
        print('done')

        # print('SdAIndividual: getting pretrain functions...'),
        # self.pretrain_fns = [ [dnn.pretraining_functions() for dnn in dnns] for dnns in self.dnns ]
        # print('done')
        # print('SdAIndividual: getting finetune functions...'),
        # self.finetune_fns = [ [dnn.build_finetune_function() for dnn in dnns] for dnns in self.dnns ]
        # print('done')
        # print('SdAIndividual: getting predict functions...'),
        # self.predict_fns = [ [dnn.build_prediction_function() for dnn in dnns] for dnns in self.dnns ]
        # print('done')

    def _make_input(self, ndata, i, j):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :param i:
        :param j:
        :return:
        '''
        return numpy.append([], [chunk[:,j,i] for chunk in ndata]).reshape((1, self.n*self.d))

    def _make_output(self, data, i, j):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :param i:
        :param j:
        :return:
        '''
        return data[:,j,i].reshape((1, self.d))

    # def _build_pretrain_functions(self, dataset):
    #     index = T.lscalar('index')
    #     corruption_level = T.iscalar('corruption')
    #     learning_rate = T.iscalar('lr')
    #
    #     costs = [[] for i in xrange(self.n_hidden_layers)]
    #     updates = [[] for i in xrange(self.n_hidden_layers)]
    #     givens={}
    #     for j in xrange(self.h):
    #         for i in xrange(self.w):
    #             cost_updates = self.dnns[j][i].get_pretraining_cost_updates(corruption_level, learning_rate)
    #             for l, cost_update in enumerate(cost_updates):
    #                 cost, update = cost_update
    #                 costs[l].append(cost)
    #                 updates[l].extend(update)
    #             givens[self.dnns[j][i].x] = self._make_input(dataset[(index-self.n):index-1], i, j)
    #
    #     fns = []
    #     for l in xrange(self.n_hidden_layers):
    #         fn = theano.function(
    #             inputs=[
    #                 index,
    #                 theano.Param(corruption_level, default=0.2),
    #                 theano.Param(learning_rate, default=0.1)
    #             ],
    #             outputs=costs[l],
    #             updates=updates[l],
    #             givens=givens,
    #             name='pretrain'
    #         )
    #         fns.append(fn)
    #
    #     return fns
    #
    # def _build_finetune_function(self, dataset):
    #     index = T.iscalar('index')
    #     i = T.iscalar('i')
    #     j = T.iscalar('j')
    #     learning_rate = T.iscalar('lr')
    #
    #     costs = [[None for i in xrange(self.w)] for j in xrange(self.h)]
    #     updates = [[None for i in xrange(self.w)] for j in xrange(self.h)]
    #     for j in xrange(self.h):
    #         for i in xrange(self.w):
    #             cost, update = self.dnns[j][i].get_finetune_cost_updates(learning_rate)
    #             costs[j][i] = cost
    #             updates[j][i] = update
    #
    #     fn = theano.function(
    #         inputs=[
    #             index,
    #             i, j,
    #             theano.Param(learning_rate, default=0.1)
    #         ],
    #         outputs=costs[j][i],
    #         updates=updates[j][i],
    #         givens={
    #             self.dnns[j][i].x: self._make_input(dataset[(index-self.n):index-1], i, j),
    #             self.dnns[j][i].y: self._make_output(dataset[index], i, j)
    #         },
    #         name='train'
    #     )
    #
    #     return fn

    def prepare(self, i, j):
        dnn = self.dnns[j][i]
        for i in xrange(len(self.sda.params)):
            self.sda.params[i].set_value(dnn.params[i].get_value(borrow=True), borrow=True)

    def _pretrain_step(self, dataset, learning_rate):
        idx = range(self.n+1,len(dataset))
        numpy.random.shuffle(idx)

        avg_cost = 0.0

        for l in xrange(self.n_hidden_layers):
            for j in xrange(self.h):
                for i in xrange(self.w):
                    self.prepare(i, j)
                    for index in idx:
                        xdata = self._make_input(dataset[(index-self.n-1):index-1], i, j)
                        cost = self.pretrain_fns[l](xdata, corruption=0.0, lr=learning_rate)
                        avg_cost += cost

        avg_cost /= self.n_hidden_layers * len(idx) * (self.w*self.h)

        return avg_cost

    def pretrain(self, dataset, epochs=100, learning_rate=0.1, patience=9):
        '''
        pretrain the model using the dataset
        :param dataset:
        :param epochs:
        :param learning_rate:
        :return:
        '''
        loop_done = False
        epoch = 0

        best_cost = numpy.inf
        best_epoch = -1
        while (epoch < epochs) and not loop_done:
            cost = self._pretrain_step(dataset, learning_rate)
            # print('  pretrain({}): cost={}'.format(epoch, cost))
            if cost < best_cost:
                best_cost = cost
                best_epoch = epoch
            elif best_epoch + patience < epoch:
                loop_done = True

            epoch = epoch + 1

        return best_cost

    def _finetune_step(self, dataset, learning_rate):
        idx = range(self.n+1,len(dataset))
        numpy.random.shuffle(idx)

        avg_cost = 0.0

        for j in xrange(self.h):
            for i in xrange(self.w):
                self.prepare(i,j)
                for index in idx:
                    xdata = self._make_input(dataset[(index-self.n-1):index-1], i, j)
                    ydata = self._make_output(dataset[index], i, j)
                    cost = self.finetune_fn(xdata, ydata, lr=learning_rate)
                    avg_cost += cost

        avg_cost /= len(idx) * (self.w*self.h)

        return avg_cost

    def finetune(self, dataset, epochs=100, learning_rate=0.1, patience=9):
        '''
        finetune the model using the dataset
        :param dataset: an array of ndarray of (d-by-h-by-w) dimention, whose size is bigger than n
        :return:
        '''
        loop_done = False
        epoch = 0

        best_cost = numpy.inf
        best_epoch = -1
        while (epoch < epochs) and not loop_done:
            cost = self._finetune_step(dataset, learning_rate)
            # print('  finetune({}): cost={}'.format(epoch, cost))
            if cost < best_cost:
                best_cost = cost
                best_epoch = epoch
            elif best_epoch + patience < epoch:
                loop_done = True

            epoch = epoch + 1

        return best_cost

    def predict(self, ndata):
        '''
        predict the next value
        :param n: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        y = numpy.zeros((self.d,self.h,self.w), dtype=theano.config.floatX)
        for j in xrange(self.h):
            for i in xrange(self.w):
                self.prepare(i,j)
                y[:,j,i] = self.predict_fn(self._make_input(ndata, i, j))[-1]

        return y