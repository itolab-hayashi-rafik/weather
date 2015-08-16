# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from base import Network, tensor5
from layer.conv_lstm import ConvLSTM
from layer.linear_regression import LinearRegression

import optimizers as O

class StackedConvLSTM(Network):
    '''
    an implementation of Stacked LSTM
    see: https://github.com/JonathanRaiman/theano_lstm/blob/master/theano_lstm/__init__.py
    '''
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input_shape=(1,28,28),
            filter_shapes=[(1,1,3,3)]
    ):
        '''
        Initialize StackedConvLSTM
        :param numpy_rng:
        :param theano_rng:

        :type input_shape: tuple or list of length 4
        :param input_shape: (batch size, num input feature maps,
                             image height, image width)

        :type filter_shapes: list of "tuple or list of length 4"
        :param filter_shapes: [(number of filters, num input feature maps,
                               filter height, filter width)]

        :type output_shape: tuple or list of length 3
        :param output_shape: (output feature maps,
                              image height, image width)
        :return:
        '''
        super(StackedConvLSTM, self).__init__()

        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.n_outs = numpy.prod(input_shape[1:])
        self.conv_lstm_layers = []
        self.params = []
        self.n_layers = len(filter_shapes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        # the input minibatch data is of shape (n_timestep, n_samples, n_feature_maps, height, width)
        self.x = tensor5('x', dtype=theano.config.floatX) # the input minibatch data
        self.mask = T.matrix('mask', dtype=theano.config.floatX)
        # the output minibatch data is of shape (n_samples, n_feature_maps, height, width)
        self.y = T.tensor4('y', dtype=theano.config.floatX)  # the regression is presented as real values
        # end-snippet-1

        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # construct LSTM layers
        self.conv_lstm_layers = []
        for i, n_hidden in enumerate(filter_shapes):
            # determine input size
            if i == 0:
                s_in = input_shape
            else:
                s_in = self.conv_lstm_layers[-1].output_shape

            # build an LSTM layer
            layer = ConvLSTM(input_shape=s_in,
                             filter_shape=filter_shapes[i],
                             activation=T.tanh,
                             prefix="ConvLSTM{}".format(i),
                             nrng=numpy_rng,
                             trng=theano_rng)
            self.conv_lstm_layers.append(layer)

            self.params.extend(layer.params)

        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.conv_lstm_layers):
                h_, c_, _ = prev_states[i], prev_states[i+1], prev_states[i+2]
                layer_out = layer.step(m, x_, h_, c_)
                _, _, x_ = layer_out # hidden, c, output
                new_states += layer_out
            return new_states

        outputs_info = []
        for i in xrange(self.n_layers):
            outputs_info.append(dict(initial=T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, self.conv_lstm_layers[i].n_in), taps=[-1])) # h_
            outputs_info.append(dict(initial=T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, self.conv_lstm_layers[i].n_in), taps=[-1])) # c_
            outputs_info.append(dict(initial=T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, self.conv_lstm_layers[i].output_shape), taps=[-1])) # o_ (x_)

        rval, updates = theano.scan(
            step,
            sequences=[self.mask, self.x],
            n_steps=n_timesteps,
            outputs_info=outputs_info, # changed: dim_proj --> self.n_ins --> hidden_layer_sizes[i]
            name="ConvLSTM_layers"
        )

        # rval には n_timestamps 分の step() の戻り値 new_states が入っている
        #assert(len(rval) == 3*self.n_layers)
        # * rval[0]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_h
        # * rval[1]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_c
        # * rval[2]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_o
        # * rval[3]: n_timesteps x n_samples x hidden_layer_sizes[1] の LSTM0_h
        # ...
        proj = rval[-1][-1]
        # In case of averaging i.e mean pooling as defined in the paper , we take all
        # the sequence of steps for all batch samples and then take a average of
        # it(sentence wise axis=0 ) and give this sum of sentences of size (16*128)
        # see: http://theano-users.narkive.com/FPNQYJIf/problem-in-understanding-lstm-code-not-able-to-understand-the-flow-of-code-http-deeplearning-net
        # proj = (proj * self.mask[:, :, None]).sum(axis=0)
        # proj = proj / self.mask.sum(axis=0)[:, None]

        # We now need to add a logistic layer on top of the Stacked LSTM
        # self.linLayer = LinearRegression(
        #     input=proj,
        #     n_in=self.conv_lstm_layers[-1].n_out,
        #     n_out=self.n_outs,
        #     activation=T.tanh,
        #     prefix="linLayer",
        #     nrng=numpy_rng,
        #     trng=theano_rng,
        # )
        #
        # self.params.extend(self.linLayer.params)

        # self.y_pred = self.linLayer.output
        self.y_pred = proj
        self.errors = (self.y_pred - self.y).norm(L=2) / n_timesteps
        self.finetune_cost = self.errors

    def build_finetune_function(self, optimizer=O.adadelta):
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        cost = self.finetune_cost
        grads = T.grad(cost, self.params)

        f_validate = theano.function([self.x, self.mask, self.y], cost)

        f_grad_shared, f_update = optimizer(learning_rate, self.params, grads,
                                            self.x, self.mask, self.y, cost)

        return (f_grad_shared, f_update, f_validate)

    def build_prediction_function(self):
        return theano.function(
            [self.x, self.mask],
            outputs=self.y_pred
        )