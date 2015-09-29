import numpy
import theano
import theano.tensor as T

from numpy.random import RandomState
from theano.tensor.shared_randomstreams import RandomStreams

from theano.sandbox import cuda

def conv2d_keepshape(input, filters, image_shape, filter_shape, subsample=(1, 1), **kargs):
    '''
    compute convolution with its output maintaining the original shape (width, height) of the input
    :param input:
    :param filters:
    :param image_shape:
    :param filter_shape:
    :param subsample:
    :param kargs:
    :return:
    '''
    if cuda.cuda_available and cuda.dnn.dnn_available():
        # cuDNN is available
        x = cuda.dnn.dnn_conv(
            img=input,
            kerns=filters,
            border_mode=(filter_shape[2]-1, filter_shape[3]-1),
            subsample=subsample,
            conv_mode='conv'
        )
    else:
        # convolve input feature maps with filters
        # the output tensor is of shape (batch size, nb filters, input_row + filter_row - 1, input_col + filter_col - 1)
        x = T.nnet.conv2d(
            input=input,
            filters=filters,
            image_shape=image_shape,
            filter_shape=filter_shape,
            border_mode='full', # zero padding the edge
            subsample=subsample,
            **kargs
        )

        # reshape x_ so that the size of output tensor matches that of the input of LSTM
        h_bound_l = filter_shape[2] / 2
        h_bound_r = -h_bound_l if filter_shape[2] % 2 == 1 else -h_bound_l+1
        w_bound_l = filter_shape[3] / 2
        w_bound_r = -w_bound_l if filter_shape[3] % 2 == 1 else -w_bound_l+1
        if h_bound_l != h_bound_r and w_bound_l != w_bound_r:
            x = x[:, :, h_bound_l:h_bound_r, w_bound_l:w_bound_r]
        elif h_bound_l != h_bound_r:
            x = x[:, :, h_bound_l:h_bound_r, :]
        elif w_bound_l != w_bound_r:
            x = x[:, :, :, w_bound_l:w_bound_r]

    return x



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