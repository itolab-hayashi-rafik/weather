# -*- coding: utf-8 -*-
from operator import itemgetter

import numpy
import theano
import theano.tensor as T

from base import Model
from network.SdA import SdA

class SdAFullyConnected(Model):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, hidden_layers_sizes=[10]):
        self.n = n
        self.d = d
        self.w = w
        self.h = h
        self.n_inputs = n*d*w*h
        self.n_hidden_layers = len(hidden_layers_sizes)
        self.n_outputs = d*w*h

        print('SdAIndividual: building the model...'),
        self.sda = SdA(
            numpy_rng,
            n_ins=self.n_inputs,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=self.n_outputs
        )
        print('done')

        print('SdAIndividual: building pretrain function...'),
        self.pretrain_fns = self.sda.pretraining_functions()
        print('done')

        print('SdAIndividual: building finetune function...'),
        self.finetune_fn, self.validate_fn = self.sda.build_finetune_function()
        print('done')

        print('SdAIndividual: building predict function...'),
        self.predict_fn = self.sda.build_prediction_function()
        print('done')

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.n) for n in idx], :].reshape((len(idx), self.n_inputs))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[n+self.n for n in idx], :].reshape((len(idx), self.n_outputs))

    def _pretrain_step(self, layer, dataset, index, corruption, learning_rate, batch_size):
        avg_cost = 0.0

        idx = range(index*batch_size, (index+1)*batch_size)
        xdata = self._make_input(dataset, idx)
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
                # print('  pretrain({0}): cost={1}'.format(epoch, cost))

                epoch = epoch + 1

        return avg_cost

    def _finetune_step(self, dataset, idx, learning_rate):
        avg_cost = 0.0

        xdata = self._make_input(dataset, idx)
        ydata = self._make_output(dataset, idx)
        cost = self.finetune_fn(xdata, ydata, lr=learning_rate)
        avg_cost += cost

        avg_cost /= len(idx) * (self.w*self.h)

        return avg_cost

    def finetune(self, dataset, train_idx, valid_idx, epochs=100, learning_rate=0.1, batch_size=1):
        '''
        finetune the model using the dataset
        :param dataset: an array of ndarray of (d-by-h-by-w) dimention, whose size is bigger than n
        :return:
        '''
        n_train_batches = len(train_idx) / batch_size

        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = numpy.inf

        done_looping = False
        epoch = 0

        avg_cost = numpy.inf
        while (epoch < epochs) and not done_looping:
            c = []
            for minibatch_index in xrange(n_train_batches):
                idx = train_idx[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
                minibatch_avg_cost = self._finetune_step(dataset, idx, learning_rate)
                c.append(minibatch_avg_cost)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                # early-stop
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = self.validate(dataset, valid_idx, batch_size)
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                                    this_validation_loss < best_validation_loss *
                                    improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

            # print('  finetune({0}): cost={1}'.format(epoch, cost))

            avg_cost = numpy.mean(c)
            epoch = epoch + 1

        return avg_cost

    def validate(self, dataset, valid_idx, batch_size):
        n_validate_batches = len(valid_idx) / batch_size

        costs = []
        for minibatch_index in xrange(n_validate_batches):
            idx = valid_idx[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
            xdata = self._make_input(dataset, idx)
            ydata = self._make_output(dataset, idx)
            costs.append(self.validate_fn(xdata, ydata))

        return costs

    def train(self, dataset, train_idx, valid_idx, epochs=100, learning_rate=0.1, batch_size=1):
        return self.finetune(dataset, train_idx, valid_idx, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    def predict(self, dataset):
        '''
        predict the next value
        :param n: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        xdata = self._make_input(dataset, [len(dataset)-self.n])
        y = self.predict_fn(xdata)[-1].reshape((self.d, self.h, self.w))

        return y