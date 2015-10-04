# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T

from base import Layer
from rnn import RNN

class LSTM(RNN):
    """
    LSTM
    see: http://deeplearning.net/tutorial/lstm.html
    see: https://github.com/JonathanRaiman/theano_lstm/blob/master/theano_lstm/__init__.py
    """
    def __init__(self, n_in, n_out, has_input=True, activation=T.tanh, clip_gradients=False, prefix="LSTM", **kwargs):
        self.has_input = has_input
        super(LSTM, self).__init__(n_in, n_out, activation=activation, clip_gradients=clip_gradients, prefix=prefix, **kwargs)

    def random_initialization(self, size):
        return (self.nrng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)

    def setup(self):
        if self.has_input:
            Wf_value = self.random_initialization((self.n_in + 2*self.n_out, self.n_out))
            self.Wf = self._shared(Wf_value, name="Wf", borrow=True)
        else:
            Wf_value = self.random_initialization((2*self.n_out, self.n_out))
            self.Wf = self._shared(Wf_value, name="Wf", borrow=True)
        bf_value = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.bf = self._shared(bf_value, name="bf", borrow=True)

        if self.has_input:
            Wi_value = self.random_initialization((self.n_in + 2*self.n_out, self.n_out))
            self.Wi = self._shared(Wi_value, name="Wi", borrow=True)
        else:
            Wi_value = self.random_initialization((2*self.n_out, self.n_out))
            self.Wi = self._shared(Wi_value, name="Wi", borrow=True)
        bi_value = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.bi = self._shared(bi_value, name="bi", borrow=True)

        if self.has_input:
            Wc_value = self.random_initialization((self.n_in + self.n_out, self.n_out))
            self.Wc = self._shared(Wc_value, name="Wc", borrow=True)
        else:
            Wc_value = self.random_initialization((self.n_out, self.n_out))
            self.Wc = self._shared(Wc_value, name="Wc", borrow=True)
        bc_value = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.bc = self._shared(bc_value, name="bc", borrow=True)

        if self.has_input:
            Wo_value = self.random_initialization((self.n_in + 2*self.n_out, self.n_out))
            self.Wo = self._shared(Wo_value, name="Wo", borrow=True)
        else:
            Wo_value = self.random_initialization((2*self.n_out, self.n_out))
            self.Wo = self._shared(Wo_value, name="Wo", borrow=True)
        bo_value = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.bo = self._shared(bo_value, name="bo", borrow=True)


    def step(self, m_, x_, c_, h_):
        # このとき x_ は _step() の外の state_below, つまり n_timestamps * n_samples * dim_proj の入力 3d tensor から
        # timestep ごとに切られた、n_samples x dim_proj の 1 タイムステップでの RNN への入力のミニバッチが入っている.
        if self.has_input:
            obs1 = T.concatenate([c_, h_, x_], axis=1)
            obs2 = T.concatenate([h_, x_], axis=1)
        else:
            obs1 = T.concatenate([c_, h_], axis=1)
            obs2 = h_

        f = T.nnet.sigmoid(T.dot(obs1, self.Wf) + self.bf)
        i = T.nnet.sigmoid(T.dot(obs1, self.Wi) + self.bi)
        c = self.activation(T.dot(obs2, self.Wc) + self.bc)
        c = f * c_ + i * c

        if self.has_input:
            obs3 = T.concatenate([c,  h_, x_], axis=1)
        else:
            obs3 = T.concatenate([c,  h_], axis=1)
        o = T.nnet.sigmoid(T.dot(obs3, self.Wo) + self.bo)
        h = o * self.activation(c)

        return c, h

    def outputs_info(self, n_samples):
        # initialize hidden states: c, h
        shape = (n_samples, self.n_out)
        return [
            T.unbroadcast(T.alloc(numpy.asarray(0., dtype=theano.config.floatX), *shape), *range(len(shape))), # c
            T.unbroadcast(T.alloc(numpy.asarray(0., dtype=theano.config.floatX), *shape), *range(len(shape)))  # h
        ]

    @property
    def params(self):
        return [self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo]

    @params.setter
    def params(self, param_list):
        self.Wf.set_value(param_list[0].get_value())
        self.bf.set_value(param_list[1].get_value())
        self.Wi.set_value(param_list[2].get_value())
        self.bi.set_value(param_list[3].get_value())
        self.Wc.set_value(param_list[4].get_value())
        self.bc.set_value(param_list[5].get_value())
        self.Wo.set_value(param_list[6].get_value())
        self.bo.set_value(param_list[7].get_value())