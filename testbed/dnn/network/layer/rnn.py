import numpy as np
import theano
import theano.tensor as T

from base import Layer

mode = theano.Mode(linker='cvm')

class RNN(Layer):
    """
    Recurrent Neural Network
    """
    def __init__(self, n_in, n_out, activation=T.tanh, clip_gradients=False, prefix="RNN", **kwargs):
        self.is_recursive = True
        super(RNN, self).__init__(None, n_in, n_out, activation=activation, clip_gradients=clip_gradients, prefix=prefix, **kwargs)

    def setup(self):
        super(RNN, self).setup()
        # hidden units
        h_value = np.zeros(shape=(self.n_out,), dtype=theano.config.floatX)
        self.h = self._shared(value=h_value, name='h')

    def step(self, *args, **kwargs):
        x = args[0]

        lin_output = T.dot(self.W, T.concatenate([self.h, x])) + self.b
        if self.activation is None:
            return lin_output
        else:
            return self.activation(lin_output)

    def outputs_info(self, *args, **kwargs):
        return []

    @property
    def output(self):
        raise NotImplementedError

class ElmanRNN(RNN):
    """
    Elman Recurrent Neural Network
    """
    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh, clip_gradients=False, **kwargs):
        self.n_hidden = n_hidden
        super(ElmanRNN, self).__init__(input, n_in, n_out, activation=activation, clip_gradients=clip_gradients, **kwargs)

    def setup(self):
        # recurrent weights as a shared variable
        W_init = np.asarray(np.random.uniform(size=(self.n_out, self.n_hidden),
                                              low=-.01, high=.01),
                            dtype=theano.config.floatX)
        self.W = self._shared(value=W_init, name='W')
        # input to hidden layer weights
        W_in_init = np.asarray(np.random.uniform(size=(self.n_in, self.n_hidden),
                                                 low=-.01, high=.01),
                               dtype=theano.config.floatX)
        self.W_in = self._shared(value=W_in_init, name='W_in')

        # hidden to output layer weights
        W_out_init = np.asarray(np.random.uniform(size=(self.n_hidden, self.n_out),
                                                  low=-.01, high=.01),
                                dtype=theano.config.floatX)
        self.W_out = self._shared(value=W_out_init, name='W_out')

        h0_init = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.h0 = self._shared(value=h0_init, name='h0')

        bh_init = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.bh = self._shared(value=bh_init, name='bh')

        by_init = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.by = self._shared(value=by_init, name='by')

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

    def step(self, x_t, h_tm1):
        h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.bh)
        y_t = T.dot(h_t, self.W_out) + self.by
        return h_t, y_t

    @property
    def params(self):
        return [self.W, self.W_in, self.W_out, self.h0, self.bh, self.by]

    @params.setter
    def params(self, param_list):
        assert len(self.params) == len(param_list)
        for i in xrange(len(self.params)):
            self.params[i].set_value(param_list[i].get_value())