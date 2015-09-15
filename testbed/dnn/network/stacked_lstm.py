# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gof.utils import flatten

from base import Network
from layer.lstm import LSTM
from layer.linear_regression import LinearRegression

import optimizers as O

class StackedLSTM(Network):
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
            n_ins=784,
            hidden_layers_sizes=[500, 500],
            n_outs=10,
            initial_hidden_states=None
    ):
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.hidden_layers_sizes = hidden_layers_sizes
        self.initial_hidden_states = initial_hidden_states
        self.lstm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0
        if self.initial_hidden_states is not None:
            assert len(self.initial_hidden_states) == self.n_layers

        self.x = input
        self.mask = mask
        self.y = None

        super(StackedLSTM, self).__init__(numpy_rng, theano_rng)

    def setup(self):
        # allocate symbolic variables for the data
        if self.x is None:
            # the input minibatch data is of shape (n_timestep, n_samples, dim_proj)
            self.x = T.tensor3('x', dtype=theano.config.floatX) # the input minibatch data
        if self.mask is None:
            # the input minibatch mask is of shape (n_timestep, n_samples)
            self.mask = T.matrix('mask', dtype=theano.config.floatX)
        if self.y is None:
            # the output minibatch data is of shape (n_samples, dim_proj)
            self.y = T.matrix('y', dtype=theano.config.floatX)  # the regression is presented as real values

        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # construct LSTM layers
        self.lstm_layers = []
        for i, n_hidden in enumerate(self.hidden_layers_sizes):
            # determine input size
            if i == 0:
                input_size = self.n_ins
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # build an LSTM layer
            layer = LSTM(n_in=input_size,
                         n_out=self.hidden_layers_sizes[i],
                         activation=T.tanh,
                         prefix="LSTM{}".format(i),
                         nrng=self.numpy_rng,
                         trng=self.theano_rng)
            self.lstm_layers.append(layer)
            self.params.extend(layer.params)

        # set initial states of layers
        if self.initial_hidden_states is not None:
            # flatten the given state list
            outputs_info = [param for layer in self.initial_hidden_states for param in layer]
        else:
            outputs_info = []
            for layer in self.lstm_layers:
                outputs_info += layer.outputs_info(n_samples)

        # feed forward calculation
        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.lstm_layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(m, x_, c_, h_)
                _, x_ = layer_out # c, h
                new_states += layer_out
            return new_states

        rval, updates = theano.scan(
            step,
            sequences=[self.mask, self.x],
            n_steps=n_timesteps,
            outputs_info=outputs_info,
            name="LSTM_layers"
        )
        self.rval = rval

        # rval には n_timestamps 分の step() の戻り値 new_states が入っている
        #assert(len(rval) == 3*self.n_layers)
        # * rval[0]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_h
        # * rval[1]: n_timesteps x n_samples x hidden_layer_sizes[0] の LSTM0_c
        # * rval[2]: n_timesteps x n_samples x hidden_layer_sizes[1] の LSTM0_h
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
                self.rval[2*i][-1],     # LSTM[i].c[T]
                self.rval[2*i+1][-1],   # LSTM[i].h[T]
            ] for i in xrange(self.n_layers)
        ]

    @property
    def params(self):
        return [[layer.params] for layer in self.lstm_layers]

    @params.setter
    def params(self, param_list):
        for layer, params in zip(self.lstm_layers, param_list):
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