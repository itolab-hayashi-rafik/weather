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
    def __init__(self, n_in, n_out, activation=T.tanh, clip_gradients=False, prefix="LSTM", **kwargs):
        super(LSTM, self).__init__(n_in, n_out, activation=activation, clip_gradients=clip_gradients, prefix=prefix, **kwargs)

    @staticmethod
    def _ortho_weight(ndim):
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype(theano.config.floatX)

    @staticmethod
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def random_initialization(self, size):
        return (self.nrng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)

    def setup(self):
        # W_value = numpy.concatenate([
        #     LSTM._ortho_weight(self.n_out),
        #     LSTM._ortho_weight(self.n_out),
        #     LSTM._ortho_weight(self.n_out),
        #     LSTM._ortho_weight(self.n_out),
        # ], axis=1)
        W_value = numpy.concatenate([
            self.random_initialization((self.n_in, self.n_out)),
            self.random_initialization((self.n_in, self.n_out)),
            self.random_initialization((self.n_in, self.n_out)),
            self.random_initialization((self.n_in, self.n_out)),
        ], axis=1)
        self.W = self._shared(W_value, name="W")

        U_value = numpy.concatenate([
            LSTM._ortho_weight(self.n_out),
            LSTM._ortho_weight(self.n_out),
            LSTM._ortho_weight(self.n_out),
            LSTM._ortho_weight(self.n_out),
        ], axis=1)
        # U_value = numpy.concatenate([
        #     self.random_initialization((self.n_in, self.n_out)),
        #     self.random_initialization((self.n_in, self.n_out)),
        #     self.random_initialization((self.n_in, self.n_out)),
        #     self.random_initialization((self.n_in, self.n_out)),
        # ], axis=1)
        self.U = self._shared(U_value, name="U")

        b_value = numpy.zeros((4 * self.n_out,), dtype=theano.config.floatX)
        self.b = self._shared(b_value, name="b")

    def step(self, m_, x_, h_, c_):
        # このとき x_ は _step() の外の state_below, つまり n_timestamps x n_samples x dim_proj の入力 3d tensor から
        # timestep ごとに切られた、n_samples x dim_proj の 1 タイムステップでの RNN への入力のミニバッチが入っている.
        # この実装では、ある条件(チュートリアル参照)を加えることで、i,f,o,c を結合(concatenate)した1つの行列での計算に簡単化している.
        preact = T.dot(h_, self.U)
        preact += (T.dot(x_, self.W) + self.b)

        i = T.nnet.sigmoid(LSTM._slice(preact, 0, self.n_out))
        f = T.nnet.sigmoid(LSTM._slice(preact, 1, self.n_out))
        o = self.activation(LSTM._slice(preact, 2, self.n_out)) # changed from T.nnet.sigmoid(...) to self.activation(...)
        c = T.tanh(LSTM._slice(preact, 3, self.n_out))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c, o

    @property
    def params(self):
        return [self.W, self.U, self.b]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        self.U.set_value(param_list[1].get_value())
        self.b.set_value(param_list[2].get_value())