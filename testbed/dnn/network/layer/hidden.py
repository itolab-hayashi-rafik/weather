import numpy
import theano
import theano.tensor as T

from base import Layer

class HiddenLayer(Layer):
    '''
    Basic hidden layer
    '''
    def __init__(self, input, n_in, n_out, activation=T.tanh, clip_gradients=False, prefix="Layer", **kwargs):
        super(HiddenLayer, self).__init__(input, n_in, n_out, activation, clip_gradients, prefix, **kwargs)

    def setup(self):
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        W_values = numpy.asarray(
            self.nrng.uniform(
                low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)
            ),
            dtype=theano.config.floatX
        )
        if self.activation == T.nnet.sigmoid:
            W_values *= 4
        self.W = self._shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = self._shared(value=b_values, name='b', borrow=True)

    @property
    def output(self):
        lin_output = T.dot(self.input, self.W) + self.b
        if self.activation is None:
            return lin_output
        else:
            return self.activation(lin_output)

    @property
    def params(self):
        return [self.W, self.b]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        self.b.set_value(param_list[1].get_value())