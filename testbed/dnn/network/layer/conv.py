# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from base import Layer


class Conv(Layer):
    def __init__(self, input, input_shape, filter_shape, poolsize=(1,1), border_mode='full', activation=T.tanh, clip_gradients=False, prefix="Conv", **kwargs):
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
        self.poolsize = poolsize
        self.border_mode = border_mode

        # LSTM receives in total:
        # "num of output feature maps * input height * input width / pooling size" inputs
        n_in = numpy.prod(input_shape[1:]) / numpy.prod(poolsize)

        # the num of output units is the same as that of input, so that the ConvLSTM in the next layer
        # can receive exactly the same number of input as this layer receives
        # FIXME: consider downsampling, using poolsize
        n_out = n_in

        super(Conv, self).__init__(input, n_in, n_out, activation, clip_gradients, prefix, **kwargs)

    def random_initialization(self, size):
        return (self.nrng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)

    def conv(self, input, filters):
        # convolve input feature maps with filters
        # the output tensor is of shape (batch size, nb filters, input_row + filter_row - 1, input_col + filter_col - 1)
        x = T.nnet.conv2d(
            input=input,
            filters=filters,
            filter_shape=self.filter_shape,
            image_shape=(None, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            border_mode=self.border_mode # zero padding the edge
        ) # (n_samples, self.filter_shape[0], self.input_shape[1] + self.filter_shape[1] - 1, self.input_shape[2] + self.filter_shape[2] - 1)

        # downsample each feature map individually, using maxpooling
        x = downsample.max_pool_2d(
            input=x,
            ds=self.poolsize,
            ignore_border=True
        )

        # reshape x_ so that the size of output tensor matches that of the input tensor
        if self.border_mode == 'full':
            h_bound_l = int(self.filter_shape[2] / 2)
            h_bound_r = -h_bound_l if self.filter_shape[2] % 2 == 1 else -h_bound_l+1
            w_bound_l = int(self.filter_shape[3] / 2)
            w_bound_r = -w_bound_l if self.filter_shape[3] % 2 == 1 else -w_bound_l+1
            if h_bound_l != h_bound_r and w_bound_l != w_bound_r:
                x = x[:, :, h_bound_l:h_bound_r, w_bound_l:w_bound_r]
            elif h_bound_l != h_bound_r:
                x = x[:, :, h_bound_l:h_bound_r, :]
            elif w_bound_l != w_bound_r:
                x = x[:, :, :, w_bound_l:w_bound_r]
        elif self.border_mode == 'valid':
            pass
            # FIXME: fill the lacking value on the border by padding zero or copying the nearest value
        else:
            raise NotImplementedError("border_mode must be either 'full' or 'valid'")

        return x

    def setup(self):
        W_value = self.random_initialization(self.filter_shape)
        self.W = self._shared(W_value, name="W", borrow=True)

    @property
    def output(self):
        return self.conv(self.input, self.W)

    @property
    def params(self):
        return [self.W]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
