# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gof.utils import flatten

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
            input=None,
            mask=None,
            input_shape=(1,28,28),
            filter_shapes=[(1,1,3,3)],
            initial_hidden_states=None
    ):
        '''
        Initialize StackedConvLSTM
        :param numpy_rng:
        :param theano_rng:

        :type input_shape: tuple or list of length 3
        :param input_shape: (num input feature maps, image height, image width)

        :type filter_shapes: list of "tuple or list of length 4"
        :param filter_shapes: [(number of filters, num input feature maps, filter height, filter width)]

        :type initial_hidden_states: list of initial hidden states
        :param initial_hidden_states: list of initial hidden states
        :return:
        '''
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.output_shape = (input_shape[0], filter_shapes[-1][0], input_shape[1], input_shape[2])
        self.initial_hidden_states = initial_hidden_states
        self.n_outs = numpy.prod(input_shape[1:])
        self.conv_lstm_layers = []
        self.n_layers = len(filter_shapes)

        assert self.n_layers > 0
        if self.initial_hidden_states is not None:
            assert len(self.initial_hidden_states) == self.n_layers

        self.x = input
        self.mask = mask
        self.y = None

        super(StackedConvLSTM, self).__init__(numpy_rng, theano_rng)

    def setup(self):
        if self.x is None:
            # allocate symbolic variables for the data
            # the input minibatch data is of shape (n_timestep, n_samples, n_feature_maps, height, width)
            self.x = tensor5('x', dtype=theano.config.floatX) # the input minibatch data
        if self.mask is None:
            # the input minibatch mask is of shape (n_timestep, n_samples, n_feature_maps)
            self.mask = T.tensor3('mask', dtype=theano.config.floatX) # FIXME: not used
        if self.y is None:
            # the output minibatch data is of shape (n_samples, n_feature_maps, height, width)
            self.y = T.tensor4('y', dtype=theano.config.floatX)  # the regression is presented as real values

        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # construct LSTM layers
        self.conv_lstm_layers = []
        for i, n_hidden in enumerate(self.filter_shapes):
            # determine input size
            if i == 0:
                s_in = self.input_shape
            else:
                s_in = self.conv_lstm_layers[-1].output_shape

            # build an LSTM layer
            layer = ConvLSTM(input_shape=s_in,
                             filter_shape=self.filter_shapes[i],
                             activation=T.tanh,
                             prefix="ConvLSTM{}".format(i),
                             nrng=self.numpy_rng,
                             trng=self.theano_rng)
            self.conv_lstm_layers.append(layer)

        # set initial states of layers
        if self.initial_hidden_states is not None:
            # flatten the given state list
            outputs_info = [param for layer in self.initial_hidden_states for param in layer]
        else:
            outputs_info = []
            for layer in self.conv_lstm_layers:
                outputs_info += layer.outputs_info(n_samples)

        # feed forward calculation
        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.conv_lstm_layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(m, x_, c_, h_)
                _, x_ = layer_out # hidden, c
                new_states += layer_out
            return new_states

        rval, updates = theano.scan(
            step,
            sequences=[self.mask, self.x],
            n_steps=n_timesteps,
            outputs_info=outputs_info, # changed: dim_proj --> self.n_ins --> hidden_layer_sizes[i]
            name="ConvLSTM_layers"
        )
        self.rval = rval

        # rval には n_timestamps 分の step() の戻り値 new_states が入っている
        #assert(len(rval) == 3*self.n_layers)
        # * rval[0]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_c
        # * rval[1]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_h
        # * rval[2]: n_timesteps x n_samples x hidden_layer_sizes[1] の LSTM0_c
        # ...

        self.finetune_cost = (self.output - self.y).norm(L=2) / n_timesteps

    @property
    def output(self):
        '''
        :return: the output of the last layer at the last time period
        '''
        return self.rval[-1][-1]

    @property
    def outputs(self):
        '''
        :return: the outputs of the last layer from time period 0 to T
        '''
        return self.rval[-1]

    @property
    def last_states(self):
        return [
            [
                self.rval[2*i][-1],     # ConvLSTM[i].c[T]
                self.rval[2*i+1][-1],   # ConvLSTM[i].h[T]
            ] for i in xrange(self.n_layers)
        ]

    @property
    def params(self):
        return [[layer.params] for layer in self.conv_lstm_layers]

    @params.setter
    def params(self, param_list):
        for layer, params in zip(self.conv_lstm_layers, param_list):
            layer.params = params

    def build_finetune_function(self, optimizer=O.adadelta):
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        cost = self.finetune_cost
        params = flatten(self.params)
        grads = T.grad(cost, params)

        f_validate = theano.function([self.x, self.mask, self.y], cost)

        f_grad_shared, f_update = optimizer(learning_rate, params, grads,
                                            self.x, self.mask, self.y, cost)

        return (f_grad_shared, f_update, f_validate)

    def build_prediction_function(self):
        return theano.function(
            [self.x, self.mask],
            outputs=self.output
        )