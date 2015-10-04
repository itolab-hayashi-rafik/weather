# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.sandbox import cuda

from base import Layer

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
    if cuda.cuda_available and cuda.dnn.dnn_available() and filter_shape[2] % 2 == 1 and filter_shape[3] % 2 == 1:
        # cuDNN is available
        x = cuda.dnn.dnn_conv(
            img=input,
            kerns=filters,
            border_mode=(filter_shape[2]//2, filter_shape[3]//2),
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
        h_shift = filter_shape[2] // 2
        w_shift = filter_shape[3] // 2
        x = x[:, :, h_shift:image_shape[2]+h_shift, w_shift:image_shape[3]+w_shift]

    return x

class Conv(Layer):
    def __init__(self, input, input_shape, filter_shape, activation=T.nnet.sigmoid, clip_gradients=False, prefix="Conv", **kwargs):
        '''
        initialize Convolutional Neural Network
        :param input:
        :param n_in:
        :param n_out:
        :param activation:
        :param clip_gradients:
        :param prefix:
        :param kwargs:
        :return:
        '''
        # assert that the number of input feature maps equals to the number of feature maps in filter_shape
        assert(input_shape[0] == filter_shape[1])

        self.input = input
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.output_shape = (self.filter_shape[0], self.input_shape[1], self.input_shape[2])

        # LSTM receives in total:
        # "num of output feature maps * input height * input width / pooling size" inputs
        n_in = numpy.prod(self.input_shape)

        # the num of output units is the same as that of input, so that the ConvLSTM in the next layer
        # can receive exactly the same number of input as this layer receives
        # FIXME: consider downsampling, using poolsize
        n_out = numpy.prod(self.output_shape)

        super(Conv, self).__init__(input, n_in, n_out, activation, clip_gradients, prefix, **kwargs)

    def random_initialization(self, size):
        return (self.nrng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)

    def setup(self):
        W_value = self.random_initialization(self.filter_shape)
        self.W = self._shared(W_value, name="W", borrow=True)
        b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = self._shared(value=b_values, name="b", borrow=True)

    @property
    def output(self):
        conv_out = conv2d_keepshape(
            input=self.input,
            filters=self.W,
            image_shape=(None, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            filter_shape=self.filter_shape
        )
        return self.activation(conv_out + self.b.dimshuffle('x',0,'x','x'))

    @property
    def params(self):
        return [self.W, self.b]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        self.b.set_value(param_list[1].get_value())
