# -*- coding: utf-8 -*-
from operator import itemgetter

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

    def _make_input(self, dataset, idx, i, j):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :param i:
        :param j:
        :return:
        '''
        # return numpy.append([], [chunk[:,j,i] for chunk in ndata]).reshape((1, self.n*self.d))
        return dataset[[range(n,n+self.n) for n in idx], :, j, i].reshape((len(idx), self.n*self.d))
        # x = []
        # for n in idx:
        #     x.append(numpy.append([], [chunk[:,j,i] for chunk in dataset[n:n+self.n]]))
        # return numpy.asarray(x, dtype=theano.config.floatX)

    def _make_output(self, dataset, idx, i, j):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :param i:
        :param j:
        :return:
        '''
        # return data[:,j,i].reshape((1, self.d))
        return dataset[[n+self.n for n in idx], :, j, i].reshape((len(idx), self.d))
        # y = []
        # for n in idx:
        #     y.append(dataset[n+self.n][:,j,i])
        # return numpy.asarray(y, dtype=theano.config.floatX)

    # def _build_pretrain_functions(self, dataset):
    #     index = T.lscalar('index')
    #     corruption_level = T.iscalar('corruption')
    #     learning_rate = T.scalar('lr')
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
    #     learning_rate = T.scalar('lr')
    #
    #     costs = []
    #     updates = []
    #     givens = {}
    #     for j in xrange(self.h):
    #         for i in xrange(self.w):
    #             cost, update = self.dnns[j][i].get_finetune_cost_updates(learning_rate)
    #             costs.append(cost)
    #             updates.extend(update)
    #             givens[self.dnns[j][i].x] = self._make_input(dataset, [1], i, j)
    #             givens[self.dnns[j][i].y] = self._make_output(dataset, [1], i, j)
    #
    #     fn = theano.function(
    #         inputs=[
    #             theano.Param(learning_rate, default=0.1)
    #         ],
    #         outputs=costs,
    #         updates=updates,
    #         givens=givens,
    #         name='train'
    #     )
    #
    #     return fn

    def prepare(self, i, j):
        dnn = self.dnns[j][i]
        for i in xrange(len(self.sda.params)):
            self.sda.params[i].set_value(dnn.params[i].get_value(borrow=True), borrow=True)

    def _pretrain_step(self, layer, dataset, index, corruption, learning_rate, batch_size):
        avg_cost = 0.0

        idx = range(index*batch_size, (index+1)*batch_size)
        for j in xrange(self.h):
            for i in xrange(self.w):
                self.prepare(i, j)
                xdata = self._make_input(dataset, idx, i, j)
                cost = self.pretrain_fns[layer](xdata, corruption=corruption, lr=learning_rate)
                avg_cost += cost

        avg_cost /= self.n_hidden_layers * (self.w*self.h)

        return avg_cost

    def pretrain(self, dataset, epochs=100, learning_rate=0.1, batch_size=1):
        '''
        pretrain the model using the dataset
        :param dataset:
        :param epochs:
        :param learning_rate:
        :return:
        '''
        n_train_batches = (len(dataset) - self.n) / batch_size

        avg_cost = numpy.inf
        for layer in xrange(self.n_hidden_layers):
            loop_done = False
            epoch = 0
            while (epoch < epochs) and not loop_done:
                c = []
                for minibatch_index in xrange(n_train_batches):
                    minibatch_avg_cost = self._pretrain_step(layer, dataset, minibatch_index, 0.0, learning_rate, batch_size)
                    c.append(minibatch_avg_cost)

                avg_cost = numpy.mean(c)
                # print('  pretrain({}): cost={}'.format(epoch, cost))

                epoch = epoch + 1

        return avg_cost

    def _finetune_step(self, dataset, index, learning_rate, batch_size):
        avg_cost = 0.0

        idx = range(index*batch_size, (index+1)*batch_size)
        for j in xrange(self.h):
            for i in xrange(self.w):
                self.prepare(i,j)
                xdata = self._make_input(dataset, idx, i, j)
                ydata = self._make_output(dataset, idx, i, j)
                cost = self.finetune_fn(xdata, ydata, lr=learning_rate)
                avg_cost += cost

        avg_cost /= len(idx) * (self.w*self.h)

        return avg_cost

    def finetune(self, dataset, epochs=100, learning_rate=0.1, batch_size=1):
        '''
        finetune the model using the dataset
        :param dataset: an array of ndarray of (d-by-h-by-w) dimention, whose size is bigger than n
        :return:
        '''
        n_train_batches = (len(dataset) - self.n) / batch_size

        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = numpy.inf

        loop_done = False
        epoch = 0

        avg_cost = numpy.inf
        while (epoch < epochs) and not loop_done:
            c = []
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = self._finetune_step(dataset, minibatch_index, learning_rate, batch_size)
                c.append(minibatch_avg_cost)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                # FIXME: early-stop 実装
                # if (iter + 1) % validation_frequency == 0:
                #     validation_losses = validate_model()
                #     this_validation_loss = numpy.mean(validation_losses)
                #     print('epoch %i, minibatch %i/%i, validation error %f %%' %
                #           (epoch, minibatch_index + 1, n_train_batches,
                #            this_validation_loss * 100.))
                #
                #     # if we got the best validation score until now
                #     if this_validation_loss < best_validation_loss:
                #
                #         #improve patience if loss improvement is good enough
                #         if (
                #                     this_validation_loss < best_validation_loss *
                #                     improvement_threshold
                #         ):
                #             patience = max(patience, iter * patience_increase)
                #
                #         # save best validation score and iteration number
                #         best_validation_loss = this_validation_loss
                #         best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

            # print('  finetune({}): cost={}'.format(epoch, cost))

            avg_cost = numpy.mean(c)
            epoch = epoch + 1

        return avg_cost

    def predict(self, dataset):
        '''
        predict the next value
        :param n: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        y = numpy.zeros((self.d,self.h,self.w), dtype=theano.config.floatX)
        for j in xrange(self.h):
            for i in xrange(self.w):
                self.prepare(i,j)
                y[:,j,i] = self.predict_fn(self._make_input(dataset, [len(dataset)-self.n], i, j))[-1]

        return y