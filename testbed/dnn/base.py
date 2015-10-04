# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy
import theano
import theano.tensor as T
from theano.gof.utils import flatten

import optimizers as O

class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_pretrain_function(self, *args, **kwargs):
        return

    @abstractmethod
    def build_finetune_function(self, *args, **kwargs):
        return

    @abstractmethod
    def build_prediction_function(self, *args, **kwargs):
        return


class BaseModel(Model):
    def __init__(self, numpy_rng, theano_rng, dnn, t_in=2, d=1, w=10, h=10, t_out=1):
        '''
        Initialize ConvLSTM Encoder-Decoder model
        :param numpy_rng:
        :param dnn: network use
        :param t_in: num of input timesteps
        :param d: input depth (num of input feature maps)
        :param w: input width
        :param h: input height
        :param t_out: num of output timesteps
        '''
        self.nrng = numpy_rng
        self.trng = theano_rng
        self.dnn = dnn
        self.t_in = t_in
        self.d = d
        self.w = w
        self.h = h
        self.t_out = t_out

    def __getstate__(self):
        return self.params

    def __setstate__(self, state):
        self.params = state

    @property
    def params(self):
        return {
            'dnn.params': self.dnn.params,
            't_in': self.t_in,
            'd': self.d,
            'w': self.w,
            'h': self.h,
            't_out': self.t_out,
        }

    @params.setter
    def params(self, param_list):
        self.dnn.params = param_list['dnn.params']
        self.t_in = param_list['t_in']
        self.d = param_list['d']
        self.w = param_list['w']
        self.h = param_list['h']
        self.t_out = param_list['t_out']

    def build_pretrain_function(self, *args, **kwargs):
        return None

    def build_finetune_function(self, optimizer=O.my_rmsprop):
        '''
        build the finetune function
        :param optimizer: an optimizer to use
        :return:
        '''
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        y = self.dnn.y # y is of shape (n_timesteps, n_samples, n_feature_maps, height, width)
        y_ = self.dnn.output # y_ is of shape (n_timesteps, n_samples, n_feature_maps, height, width)

        n_samples = y.shape[1]

        mse = T.mean((y - y_)**2) # Mean Square Error
        cee = T.sum(T.nnet.binary_crossentropy(y_, y)) / n_samples # Cross Entropy Error
        cost = cee
        params = flatten(self.dnn.params)
        grads = T.grad(cost, params)

        f_grad_shared, f_update = optimizer(learning_rate, params, grads,
                                            self.dnn.x, self.dnn.mask, self.dnn.y, cost)

        return (f_grad_shared, f_update)

    def build_prediction_function(self):
        if self.dnn.is_rnn:
            return theano.function([self.dnn.x, self.dnn.mask], outputs=self.dnn.outputs)
        else:
            return theano.function([self.dnn.x, self.dnn.mask], outputs=self.dnn.output)

    def prepare_data(self, xs, ys, maxlen=None):
        '''
        prepare data for inserting to RNN or LSTM
        see: /lstm/tutorial/imdb.py
        :param xs:
        :param ys:
        :param maxlen:
        :return: x, mask, y
        '''
        lengths = [len(s) for s in xs]

        if maxlen is not None:
            new_seqs = []
            new_labels = []
            new_lengths = []
            for l, s, y in zip(lengths, xs, ys):
                if l < maxlen:
                    new_seqs.append(s)
                    new_labels.append(y)
                    new_lengths.append(l)
            lengths = new_lengths
            ys = new_labels
            xs = new_seqs

            if len(lengths) < 1:
                return None, None, None

        n_samples = len(xs)
        maxlen = numpy.max(lengths) # n_timesteps

        x = numpy.zeros((maxlen, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
        x_mask = numpy.zeros((maxlen, n_samples, self.d), dtype=theano.config.floatX)
        for idx, xi in enumerate(xs):
            x[:lengths[idx], idx, :, :, :] = xi
            x_mask[:lengths[idx], idx, :] = 1.

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :, :, :] = yi
        else:
            y = None

        return x, x_mask, y
