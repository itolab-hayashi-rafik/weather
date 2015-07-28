import numpy as np
import theano
import theano.tensor as T

import base

mode = theano.Mode(linker='cvm')

class RNN(base.Layer):
    """
    Recurrent Neural Network
    """
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.is_recursive = True

    def setup(self):
        super(RNN, self).setup()
        # hidden units
        h_value = np.zeros(shape=(self.n_out,), dtype=theano.config.floatX)
        self.h = theano.shared(value=h_value, name='h')

    @property
    def output(self):
        lin_output = T.dot(self.W, T.concatenate([self.h, self.input])) + self.b
        if self.activation is None:
            return lin_output
        else:
            return self.activation(lin_output)

class ElmanRNN(RNN):
    """
    Elman Recurrent Neural Network
    """
    def __init__(self, *args, **kwargs):
        super(ElmanRNN, self).__init__(*args, **kwargs)
        self.n_hidden = kwargs.get('n_hidden')

    def setup(self):
        # recurrent weights as a shared variable
        W_init = np.asarray(np.random.uniform(size=(self.n_out, self.n_hidden),
                                              low=-.01, high=.01),
                            dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W')
        # input to hidden layer weights
        W_in_init = np.asarray(np.random.uniform(size=(self.n_in, self.n_hidden),
                                                 low=-.01, high=.01),
                               dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init, name='W_in')

        # hidden to output layer weights
        W_out_init = np.asarray(np.random.uniform(size=(self.n_hidden, self.n_out),
                                                  low=-.01, high=.01),
                                dtype=theano.config.floatX)
        self.W_out = theano.shared(value=W_out_init, name='W_out')

        h0_init = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name='h0')

        bh_init = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name='by')

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        self.L1 += abs(self.W_out.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

    def errors(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    @property
    def output(self):
        def step(x_t, h_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        [self.h, self.y_pred], _ = theano.scan(step,
                                               sequences=self.input,
                                               outputs_info=[self.h0, None])

        return self.y_pred

    @property
    def params(self):
        return [self.W, self.W_in, self.W_out, self.h0, self.bh, self.by]

    @params.setter
    def params(self, param_list):
        assert len(self.params) == len(param_list)
        for i in xrange(len(self.params)):
            self.params[i].set_value(param_list[i].get_value())