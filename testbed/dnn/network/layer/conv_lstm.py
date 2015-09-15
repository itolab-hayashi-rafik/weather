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

        super(ConvLSTM, self).__init__(n_in, n_out, activation=activation, clip_gradients=clip_gradients, prefix=prefix, **kwargs)

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

        # reshape x_ so that the size of output tensor matches that of the input of LSTM
        if self.border_mode == 'full':
            h_bound_l = int(self.filter_shape[2] / 2)
            h_bound_r = -h_bound_l if self.filter_shape[2] % 2 == 1 else -h_bound_l+1
            w_bound_l = int(self.filter_shape[3] / 2)
            w_bound_r = -w_bound_l if self.filter_shape[3] % 2 == 1 else -w_bound_l+1
            x = x[:, :, h_bound_l:h_bound_r, w_bound_l:w_bound_r]
        elif self.border_mode == 'valid':
            pass
            # FIXME: fill the lacking value on the border by padding zero or copying the nearest value
        else:
            raise NotImplementedError("border_mode must be either 'full' or 'valid'")

        return x

    def setup(self):
        Wxf_value = self.random_initialization(self.filter_shape)
        self.Wxf = self._shared(Wxf_value, name="Wxf", borrow=True)
        Whf_value = self.random_initialization(self.filter_shape)
        self.Whf = self._shared(Whf_value, name="Whf", borrow=True)
        Wcf_value = self.random_initialization(self.input_shape)
        self.Wcf = self._shared(Wcf_value, name="Wcf", borrow=True)
        bf_value = numpy.zeros(self.output_shape, dtype=theano.config.floatX)
        self.bf = self._shared(bf_value, name="bf", borrow=True)

        Wxi_value = self.random_initialization(self.filter_shape)
        self.Wxi = self._shared(Wxi_value, name="Wxi", borrow=True)
        Whi_value = self.random_initialization(self.filter_shape)
        self.Whi = self._shared(Whi_value, name="Whi", borrow=True)
        Wci_value = self.random_initialization(self.input_shape)
        self.Wci = self._shared(Wci_value, name="Wci", borrow=True)
        bi_value = numpy.zeros(self.output_shape, dtype=theano.config.floatX)
        self.bi = self._shared(bi_value, name="bi", borrow=True)

        Wxc_value = self.random_initialization(self.filter_shape)
        self.Wxc = self._shared(Wxc_value, name="Wxc", borrow=True)
        Whc_value = self.random_initialization(self.filter_shape)
        self.Whc = self._shared(Whc_value, name="Whc", borrow=True)
        bc_value = numpy.zeros(self.output_shape, dtype=theano.config.floatX)
        self.bc = self._shared(bc_value, name="bc", borrow=True)

        Wxo_value = self.random_initialization(self.filter_shape)
        self.Wxo = self._shared(Wxo_value, name="Wxo", borrow=True)
        Who_value = self.random_initialization(self.filter_shape)
        self.Who = self._shared(Who_value, name="Who", borrow=True)
        Wco_value = self.random_initialization(self.input_shape)
        self.Wco = self._shared(Wco_value, name="Wco", borrow=True)
        bo_value = numpy.zeros(self.output_shape, dtype=theano.config.floatX)
        self.bo = self._shared(bo_value, name="bo", borrow=True)

    def step(self, m_, x_, c_, h_):
        # このとき x_ は _step() の外の state_below, つまり n_timestamps * n_samples * dim_proj の入力 3d tensor から
        # timestep ごとに切られた、n_samples x dim_proj の 1 タイムステップでの RNN への入力のミニバッチが入っている.

        f = T.nnet.sigmoid(self.conv(x_, self.Wxf) + self.conv(h_, self.Whf) + self.Wcf * c_ + self.bf)
        i = T.nnet.sigmoid(self.conv(x_, self.Wxi) + self.conv(h_, self.Whi) + self.Wci * c_ + self.bi)
        o = T.nnet.sigmoid(self.conv(x_, self.Wxo) + self.conv(h_, self.Who) + self.Wco * c_ + self.bo)
        c = T.tanh(self.conv(x_, self.Wxc) + self.conv(h_, self.Whc) + self.bc)

        c = f * c_ + i * c

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

class SimpleConvLSTM(RNN):
    """
    ConvLSTM
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
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.output_shape = input_shape
        self.poolsize = poolsize
        self.border_mode = border_mode

        # LSTM receives in total:
        # "num of output feature maps * input height * input width / pooling size" inputs
        n_in = numpy.prod(input_shape[1:]) / numpy.prod(poolsize)

        # the num of output units is the same as that of input, so that the ConvLSTM in the next layer
        # can receive exactly the same number of input as this layer receives
        # FIXME: consider downsampling, using poolsize
        n_out = n_in

        super(SimpleConvLSTM, self).__init__(n_in, n_out, activation=activation, clip_gradients=clip_gradients, prefix=prefix, **kwargs)

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
        #
        ## Conv
        ##  refer to /cnn/tutorial/convolutional_mlp.py
        #

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(self.filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        fan_out = (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) / numpy.prod(self.poolsize))

        # initialize weights with random weights
        Conv_W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.Conv_W = self._shared(
            numpy.asarray(
                self.nrng.uniform(low=-Conv_W_bound, high=Conv_W_bound, size=self.filter_shape),
                dtype=theano.config.floatX
            ),
            name="Conv_W",
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        Conv_b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.Conv_b = self._shared(value=Conv_b_values, name="Conv_b", borrow=True)


        #
        ## LSTM
        ##  see more details in lstm.py
        #

        LSTM_W_value = numpy.concatenate([
            self.random_initialization((self.n_in, self.n_out)),
            self.random_initialization((self.n_in, self.n_out)),
            self.random_initialization((self.n_in, self.n_out)),
            self.random_initialization((self.n_in, self.n_out)),
        ], axis=1)
        self.LSTM_W = self._shared(LSTM_W_value, name="LSTM_W", borrow=True)

        LSTM_U_value = numpy.concatenate([
            ConvLSTM._ortho_weight(self.n_out),
            ConvLSTM._ortho_weight(self.n_out),
            ConvLSTM._ortho_weight(self.n_out),
            ConvLSTM._ortho_weight(self.n_out),
        ], axis=1)
        self.LSTM_U = self._shared(LSTM_U_value, name="LSTM_U", borrow=True)

        LSTM_b_value = numpy.zeros((4 * self.n_out,), dtype=theano.config.floatX)
        self.LSTM_b = self._shared(LSTM_b_value, name="LSTM_b", borrow=True)

    def step(self, m_, x_, h_, c_):
        # x_ should be of shape (batch size, nb feature maps, input height, input width)
        n_samples = x_.shape[0]

        # convolve input feature maps with filters
        # the output tensor is of shape (batch size, nb filters, input_row + filter_row - 1, input_col + filter_col - 1)
        x_ = T.nnet.conv2d(
            input=x_,
            filters=self.Conv_W,
            filter_shape=self.filter_shape,
            image_shape=(None, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            border_mode=self.border_mode # zero padding the edge
        ) # (n_samples, self.filter_shape[0], self.input_shape[1] + self.filter_shape[1] - 1, self.input_shape[2] + self.filter_shape[2] - 1)

        # downsample each feature map individually, using maxpooling
        x_ = downsample.max_pool_2d(
            input=x_,
            ds=self.poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        x_ = T.tanh(x_ + self.Conv_b.dimshuffle('x', 0, 'x', 'x'))

        # reshape x_ so that the size of output tensor matches that of the input of LSTM
        if self.border_mode == 'full':
            h_bound = self.filter_shape[2] / 2
            w_bound = self.filter_shape[3] / 2
            x_ = x_[:, :, h_bound:-h_bound, w_bound:-w_bound]
        elif self.border_mode == 'valid':
            pass
            # FIXME: fill the lacking value on the border by padding zero or copying the nearest value
        else:
            raise NotImplementedError("border_mode must be either 'full' or 'valid'")

        # at this point, the tensor x_ is shape of (n_samples, nb filters, input height, input width)
        # we concatenate the values through filters by calculating their sum
        # to make its shape (n_samples, input height, input width) # fiXME:
        # x_ = T.sum(x_, axis=1)

        # we flatten this tensor to (n_samples, self.n_in), which is (n_samples, num of output feature maps * height * width)
        x_ = x_.reshape((n_samples, self.n_in))

        # このとき x_ は _step() の外の state_below, つまり n_timestamps * n_samples * self.n_in の入力 3d tensor から
        # timestep ごとに切られた、n_samples * self.n_in の 1 タイムステップでの RNN への入力のミニバッチが入っている.
        # この実装では、ある条件(チュートリアル参照)を加えることで、i,f,o,c を結合(concatenate)した1つの行列での計算に簡単化している.
        preact = T.dot(h_, self.LSTM_U)
        preact += (T.dot(x_, self.LSTM_W) + self.LSTM_b) # FIXME: not yet considered well. maybe ndim does not match, need to fix this

        i = T.nnet.sigmoid(ConvLSTM._slice(preact, 0, self.n_out))
        f = T.nnet.sigmoid(ConvLSTM._slice(preact, 1, self.n_out))
        o = self.activation(ConvLSTM._slice(preact, 2, self.n_out)) # changed from T.nnet.sigmoid(...) to self.activation(...)
        c = T.tanh(ConvLSTM._slice(preact, 3, self.n_out))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        # reshape the output o to make it match the output of this ConvLSTM layer, (n_samples, n_feature_maps, height, width)
        o = o.reshape((n_samples, self.input_shape[0], self.input_shape[1], self.input_shape[2])) # FIXME: reshaping from (n_samples, height * width) to (n_samples, n_feature_maps, height, width)
        o = T.unbroadcast(o, 1)

        return h, c, o

    @property
    def params(self):
        return [self.Conv_W, self.Conv_b, self.LSTM_W, self.LSTM_U, self.LSTM_b]

    @params.setter
    def params(self, param_list):
        self.Conv_W.set_value(param_list[0].get_value())
        self.Conv_b.set_value(param_list[1].get_value())
        self.LSTM_W.set_value(param_list[2].get_value())
        self.LSTM_U.set_value(param_list[3].get_value())
        self.LSTM_b.set_value(param_list[4].get_value())
