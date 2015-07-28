import numpy
import theano
import theano.tensor as T

from numpy.random import RandomState
from theano.tensor.shared_randomstreams import RandomStreams

class Layer(object):
    def __init__(self, input, n_in, n_out, activation=T.tanh, **kwargs):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer

        :type nrng: numpy.random.RandomState
        :param nrng: a random number generator used to initialize weights

        :type trng: theano.tensor.shared_randomstreams.RandomState
        :param trng: a theano random number generator
        """
        # end-snippet-1

        self.nrng = kwargs.get('nrng', RandomState(23455))
        self.trng = kwargs.get('trng', RandomStreams(self.nrng.randint(2 ** 30)))
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        # setup variables
        self.setup()

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
        if self.activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

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