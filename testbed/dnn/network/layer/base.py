import numpy
import theano
import theano.tensor as T

from numpy.random import RandomState
from theano.tensor.shared_randomstreams import RandomStreams

class Layer(object):
    def __init__(self, input, n_in, n_out, activation=T.tanh, clip_gradients=False, prefix="Layer", **kwargs):
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
        self.clip_gradients = clip_gradients
        self.prefix = prefix

        # setup variables
        self.setup()

    def _p(self, name):
        return '{0}_{1}'.format(self.prefix, name)

    def _shared(self, value, name=None, strict=False, allow_downcast=False, **kwargs):
        name = self._p(name) if name is not None else name
        return theano.shared(value, name=name, strict=strict, allow_downcast=allow_downcast, **kwargs)

    def setup(self):
        pass

    @property
    def output(self):
        raise NotImplementedError

    @property
    def params(self):
        return []

    @params.setter
    def params(self, param_list):
        pass