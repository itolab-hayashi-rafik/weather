# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gof.utils import flatten

from base import Network, tensor5
from stacked_lstm import StackedLSTM

import optimizers as O

class EncoderDecoderLSTM(Network):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            n_ins=784,
            hidden_layers_sizes=[500, 500],
            n_outs=10,
    ):
        '''
        Initialize LSTM Encoder-Decoder Network

        :return:
        '''
        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_outs = n_outs
        self.n_layers = len(hidden_layers_sizes)

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, theano_rng)

    def setup(self):
        # allocate symbolic variables for the data
        # the input minibatch data is of shape (n_timestep, n_samples, n_feature_maps, height, width)
        self.x = tensor5('x', dtype=theano.config.floatX) # the input minibatch data
        # the input minibatch mask is of shape (n_timestep, n_samples, n_feature_maps)
        self.mask = T.tensor3('mask', dtype=theano.config.floatX) # FIXME: not used
        # the output minibatch data is of shape (n_samples, n_feature_maps, height, width)
        self.y = T.tensor4('y', dtype=theano.config.floatX)  # the regression is presented as real values

        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # Encoder network
        self.encoder = StackedLSTM(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask,
            n_ins=self.n_ins,
            hidden_layers_sizes=self.hidden_layers_sizes,
            n_outs=self.n_outs
        )

        # Decoder network
        self.decoder = StackedLSTM(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.encoder.outputs, # FIXME: outputs? use beam search?
            mask=self.mask, # FIXME: is this ok?
            n_ins=self.n_outs,
            hidden_layers_sizes=[s for s in reversed(self.hidden_layers_sizes)],
            initial_hidden_states=[s for s in reversed(self.encoder.last_states)],
            n_outs=self.n_ins
        )

        # calculate the cost
        self.finetune_cost = (self.output - self.y).norm(L=2) / n_timesteps

    @property
    def output(self):
        return self.decoder.output

    def outputs(self):
        return self.decoder.outputs

    @property
    def params(self):
        return [
            self.encoder.params,
            self.decoder.params
        ]

    @params.setter
    def params(self, param_list):
        self.encoder.params = param_list[0]
        self.decoder.params = param_list[1]

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