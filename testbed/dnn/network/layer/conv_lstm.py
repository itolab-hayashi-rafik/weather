# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from rnn import RNN


class ConvLSTM(RNN):
    """
    LSTM
    see: http://deeplearning.net/tutorial/lstm.html
    see: https://github.com/JonathanRaiman/theano_lstm/blob/master/theano_lstm/__init__.py
    """
    def __init__(self, input_shape, filter_shape, poolsize=(1,1), border_mode='full', activation=T.tanh, clip_gradients=False, prefix="ConvLSTM", **kwargs):
        '''
         initialize ConvLSTM

         :type input_shape: tuple or list of length 3
         :param input_shape: (num input feature maps,
                              image height, image width)

         :type filter_shape: tuple or list of length 4
         :param filter_shape: (number of filters, num input feature maps,
                               filter height, filter width)

         :type poolsize: tuple of length 2
         :param poolsize: poolsize

         :type border_mode: String
         :param border_mode: 'full' or 'valid'

         :param activation:
         :param clip_gradients:
         :param prefix:
         :param kwargs:
         :return:
         '''
        # assert that the number of input feature maps equals to the number of feature maps in filter_shape
        assert(input_shape[0] == filter_shape[1])

        self.input_shape = input_shape
        self.input_filter_shape = filter_shape
        self.hidden_filter_shape = (filter_shape[0], filter_shape[0], filter_shape[2], filter_shape[3])
        self.output_shape = (filter_shape[0], input_shape[1], input_shape[2])
        self.poolsize = poolsize
        self.border_mode = border_mode

        # LSTM receives in total:
        # "num of output feature maps * input height * input width / pooling size" inputs
        n_in = numpy.prod(input_shape[1:]) / numpy.prod(poolsize)

        # the num of output units is the same as that of input, so that the ConvLSTM in the next layer
        # can receive exactly the same number of input as this layer receives
        # FIXME: consider downsampling, using poolsize
        n_out = n_in

        super(ConvLSTM, self).__init__(n_in, n_out, activation=activation, clip_gradients=clip_gradients, prefix=prefix, **kwargs)

    def random_initialization(self, size):
        return (self.nrng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)

    def conv(self, input, filters):
        # convolve input feature maps with filters
        # the output tensor is of shape (batch size, nb filters, input_row + filter_row - 1, input_col + filter_col - 1)
        x = T.nnet.conv2d(
            input=input,
            filters=filters,
            filter_shape=self.input_filter_shape,
            image_shape=(None, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            border_mode=self.border_mode # zero padding the edge
        )

        # downsample each feature map individually, using maxpooling
        x = downsample.max_pool_2d(
            input=x,
            ds=self.poolsize,
            ignore_border=True
        )

        # reshape x_ so that the size of output tensor matches that of the input of LSTM
        if self.border_mode == 'full':
            h_bound_l = int(self.input_filter_shape[2] / 2)
            h_bound_r = -h_bound_l if self.input_filter_shape[2] % 2 == 1 else -h_bound_l+1
            w_bound_l = int(self.input_filter_shape[3] / 2)
            w_bound_r = -w_bound_l if self.input_filter_shape[3] % 2 == 1 else -w_bound_l+1
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
        Wxf_value = self.random_initialization(self.input_filter_shape)
        self.Wxf = self._shared(Wxf_value, name="Wxf", borrow=True)
        Whf_value = self.random_initialization(self.hidden_filter_shape)
        self.Whf = self._shared(Whf_value, name="Whf", borrow=True)
        Wcf_value = self.random_initialization(self.output_shape)
        self.Wcf = self._shared(Wcf_value, name="Wcf", borrow=True)
        bf_value = numpy.zeros((self.output_shape[0],), dtype=theano.config.floatX)
        self.bf = self._shared(bf_value, name="bf", borrow=True)

        Wxi_value = self.random_initialization(self.input_filter_shape)
        self.Wxi = self._shared(Wxi_value, name="Wxi", borrow=True)
        Whi_value = self.random_initialization(self.hidden_filter_shape)
        self.Whi = self._shared(Whi_value, name="Whi", borrow=True)
        Wci_value = self.random_initialization(self.output_shape)
        self.Wci = self._shared(Wci_value, name="Wci", borrow=True)
        bi_value = numpy.zeros((self.output_shape[0],), dtype=theano.config.floatX)
        self.bi = self._shared(bi_value, name="bi", borrow=True)

        Wxc_value = self.random_initialization(self.input_filter_shape)
        self.Wxc = self._shared(Wxc_value, name="Wxc", borrow=True)
        Whc_value = self.random_initialization(self.hidden_filter_shape)
        self.Whc = self._shared(Whc_value, name="Whc", borrow=True)
        bc_value = numpy.zeros((self.output_shape[0],), dtype=theano.config.floatX)
        self.bc = self._shared(bc_value, name="bc", borrow=True)

        Wxo_value = self.random_initialization(self.input_filter_shape)
        self.Wxo = self._shared(Wxo_value, name="Wxo", borrow=True)
        Who_value = self.random_initialization(self.hidden_filter_shape)
        self.Who = self._shared(Who_value, name="Who", borrow=True)
        Wco_value = self.random_initialization(self.output_shape)
        self.Wco = self._shared(Wco_value, name="Wco", borrow=True)
        bo_value = numpy.zeros((self.output_shape[0],), dtype=theano.config.floatX)
        self.bo = self._shared(bo_value, name="bo", borrow=True)

    def step(self, m_, x_, c_, h_):
        # このとき x_ は _step() の外の state_below, つまり n_timestamps * n_samples * dim_proj の入力 3d tensor から
        # timestep ごとに切られた、n_samples x dim_proj の 1 タイムステップでの RNN への入力のミニバッチが入っている.

        f = T.nnet.sigmoid(self.conv(x_, self.Wxf) + self.conv(h_, self.Whf) + self.Wcf * c_ + self.bf.dimshuffle('x',0,'x','x'))
        i = T.nnet.sigmoid(self.conv(x_, self.Wxi) + self.conv(h_, self.Whi) + self.Wci * c_ + self.bi.dimshuffle('x',0,'x','x'))
        c = T.tanh(self.conv(x_, self.Wxc) + self.conv(h_, self.Whc) + self.bc.dimshuffle('x',0,'x','x'))
        c = f * c_ + i * c

        o = T.nnet.sigmoid(self.conv(x_, self.Wxo) + self.conv(h_, self.Who) + self.Wco * c  + self.bo.dimshuffle('x',0,'x','x'))
        h = o * T.tanh(c)

        return c, h

    def outputs_info(self, n_samples):
        return [
            dict(initial=T.patternbroadcast(T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, *self.output_shape), [False, False, False, False]), taps=[-1]), # c
            dict(initial=T.patternbroadcast(T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, *self.output_shape), [False, False, False, False]), taps=[-1]), # h
        ]

    @property
    def params(self):
        return [self.Wxf, self.Whf, self.Wcf, self.bf,
                self.Wxi, self.Whi, self.Wci, self.bi,
                self.Wxc, self.Whc, self.bc,
                self.Wxo, self.Who, self.Wco, self.bo]

    @params.setter
    def params(self, param_list):
        self.Wxf.set_value(param_list[0].get_value())
        self.Whf.set_value(param_list[1].get_value())
        self.Wcf.set_value(param_list[2].get_value())
        self.bf.set_value(param_list[3].get_value())
        self.Wxi.set_value(param_list[4].get_value())
        self.Whi.set_value(param_list[5].get_value())
        self.Wci.set_value(param_list[6].get_value())
        self.bi.set_value(param_list[7].get_value())
        self.Wxc.set_value(param_list[8].get_value())
        self.Whc.set_value(param_list[9].get_value())
        self.bc.set_value(param_list[10].get_value())
        self.Wxo.set_value(param_list[11].get_value())
        self.Who.set_value(param_list[12].get_value())
        self.Wco.set_value(param_list[13].get_value())
        self.bo.set_value(param_list[14].get_value())

