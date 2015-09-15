# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gof.utils import flatten

from base import Network, tensor5
# from stacked_conv_lstm import StackedConvLSTM
from stacked_networks import StackedConvLSTMEncoder, StackedConvLSTMDecoder

import optimizers as O

class EncoderDecoderConvLSTM(Network):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input_shape=(1,28,28),
            filter_shapes=[(1,1,3,3)]
    ):
        '''
        Initialize ConvLSTM Encoder-Decoder Network

        :type numpy_rng: numpy.random
        :param numpy_rng: numpy random

        :type input_shape: tuple or list of length 3
        :param input_shape: (num input feature maps, input height, input width)

        :type filter_shapes: list of "tuple or list of length 4"
        :param filter_shapes: [(number of filters, num input feature maps, filter height, filter width)]

        :return:
        '''
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.n_layers = len(filter_shapes)

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, theano_rng)

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
        # self.encoder = StackedConvLSTM(
        #     self.numpy_rng,
        #     theano_rng=self.theano_rng,
        #     input=self.x,
        #     mask=self.mask,
        #     input_shape=self.input_shape,
        #     filter_shapes=self.filter_shapes
        # )
        self.encoder = StackedConvLSTMEncoder(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask,
            output=self.y,
            input_shape=self.input_shape,
            filter_shapes=self.filter_shapes
        )

        # # Decoder network
        # self.decoder = StackedConvLSTM(
        #     self.numpy_rng,
        #     theano_rng=self.theano_rng,
        #     input=self.x[-1].dimshuffle('x',0,1,2,3), # FIXME: input should be [x[-1], y[0], ..., y[T']] which requires recursive input of the output of decoder network
        #     mask=self.mask, # FIXME: is this ok?
        #     input_shape=self.input_shape,
        #     filter_shapes=[s for s in reversed(self.filter_shapes)],
        #     initial_hidden_states=[s for s in reversed(self.encoder.last_states)]
        # )
        self.decoder = StackedConvLSTMDecoder(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask, # FIXME: is this ok?
            output=self.y,
            encoder=self.encoder,
            n_timesteps=2 # FIXME
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