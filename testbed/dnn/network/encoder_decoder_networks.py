# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T

from base import StandaloneNetwork, tensor5
from stacked_networks import StackedLSTM, StackedConvLSTM
from layer import Conv

class EncoderDecoderNetwork(StandaloneNetwork):
    '''
    Base implementation of Stacked Network
    '''
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="EncoderDecoderNetwork",
                 input=None,
                 mask=None,
                 output=None,
                 is_rnn=False
                 ):
        self.encoder = None
        self.decoder = None

        assert input is not None
        assert output is not None

        super(EncoderDecoderNetwork, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn)

    def setup(self):
        '''
        Construct the stacked network
        :return:
        '''
        raise NotImplementedError

    @property
    def output(self):
        return self.outputs[-1]

    @property
    def outputs(self):
        '''
        output [z_0, ..., z_T] in the following diagram

            *          *         z_0        z_1        z_2           z_T         *
            ^          ^          ^          ^          ^             ^          ^
            |          |          |          |          |             |          |
        [Encoder]->[Encoder]->[Encoder]->[Decoder]->[Decoder]...->[Decoder]->[Decoder]
            ^          ^          ^          ^          ^             ^          ^
            |          |          |          |          |             |          |
           x_0        x_1        x_2        z_0        z_1         z_(T-1)      z_T

        :return: [z_0, ..., z_T]
        '''
        n_input_timesteps = self.x.shape[0]
        return T.concatenate([self.encoder.outputs, self.decoder.outputs], axis=0)[n_input_timesteps-1:-1]

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


class EncoderDecoderLSTM(EncoderDecoderNetwork):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="EncoderDecoderLSTM",
                 input=None,
                 mask=None,
                 output=None,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 n_timesteps=1
    ):
        '''

        :param numpy_rng:
        :param theano_rng:
        :param input: input tensor of shape (n_timesteps, n_samples, n_ins)
        :param mask: input mask of shape (n_timesteps, n_samples)
        :param output: output tensor of shape (n_timesteps, n_samples, n_ins)
        :param n_ins:
        :param hidden_layers_sizes:
        :param n_timesteps: num of output timesteps
        :return:
        '''
        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_timesteps = n_timesteps

        # Allocate symbolic variables for the data
        if input is None:
            # the input minibatch data is of shape (n_timesteps, n_samples, n_ins)
            input = T.tensor3('x', dtype=theano.config.floatX)
        if mask is None:
            # the input minibatch mask is of shape (n_samples, n_ins)
            mask = T.matrix('mask', dtype=theano.config.floatX) # FIXME: not used
        if output is None:
            # the output minibatch data is of shape (n_timesteps, n_samples, n_ins)
            output = T.tensor3('y', dtype=theano.config.floatX)

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # Encoder network
        self.encoder = StackedLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedLSTMEncoder",
            input=self.x,
            mask=self.mask,
            output=self.y,
            n_ins=self.n_ins,
            hidden_layers_sizes=self.hidden_layers_sizes,
        )

        # Decoder network
        self.decoder = StackedLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedLSTMDecoder",
            output=self.y,
            hidden_layers_sizes=self.hidden_layers_sizes,
            n_timesteps=self.n_timesteps,
            initial_hidden_states=self.encoder.last_states,
            has_input=False
        )


class EncoderDecoderConvLSTM(EncoderDecoderNetwork):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="EncoderDecoderConvLSTM",
                 input=None,
                 mask=None,
                 output=None,
                 input_shape=(1,28,28),
                 filter_shapes=[(1,1,3,3)],
                 n_timesteps=1
    ):
        '''

        :param numpy_rng:
        :param theano_rng:
        :param input: input 5D tensor of shape (n_timesteps, n_samples, n_feature_maps, height, width)
        :param mask: input mask of shape (n_timesteps, n_samples, n_feature_maps)
        :param output: output 5D tensor of shape (n_timestamps, n_samples, n_feature_maps, height, width)
        :param input_shape: (num input feature maps, input height, input width)
        :param filter_shapes: [(number of filters, num input feature maps, filter height, filter width)]
        :param num of output timesteps
        :return:
        '''
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.n_timesteps = n_timesteps

        # determine conv filter shape
        n_hiddens = sum([s[0] for s in self.filter_shapes]) # the number of total output feature maps (num of hidden states)
        self.conv_input_shape = (n_hiddens, input_shape[1], input_shape[2])
        self.conv_filter_shape = (input_shape[0], n_hiddens, 1, 1)

        # Allocate symbolic variables for the data
        if input is None:
            # the input minibatch data is of shape (n_timesteps, n_samples, n_feature_maps, height, width)
            input = tensor5('x', dtype=theano.config.floatX)
        if mask is None:
            # the input minibatch mask is of shape (n_timesteps, n_samples, n_feature_maps)
            mask = T.tensor3('mask', dtype=theano.config.floatX) # FIXME: not used
        if output is None:
            # the output minibatch data is of shape (n_timesteps, n_samples, n_feature_maps, height, width)
            output = tensor5('y', dtype=theano.config.floatX)

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # Encoder network
        self.encoder = StackedConvLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedConvLSTMEncoder",
            input=self.x,
            mask=self.mask,
            output=self.y,
            input_shape=self.input_shape,
            filter_shapes=self.filter_shapes
        )

        # Decoder network
        self.decoder = StackedConvLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedConvLSTMDecoder",
            output=self.y,
            input_shape=self.input_shape,
            filter_shapes=self.filter_shapes,
            n_timesteps=self.n_timesteps,
            initial_hidden_states=self.encoder.last_states,
            has_input=False
        )

        # Conv(1x1) layer
        self.conv_layer = Conv(
            None,
            self.conv_input_shape,
            self.conv_filter_shape,
            prefix="{0}_ConvLayer".format(self.name)
        )

        self.setup_scan()

    def setup_scan(self):
        '''
        output [z_0, ..., z_T] in the following diagram

                                 z_0        z_1        z_2           z_T
                                  ^          ^          ^             ^
                                  |          |          |             |
                             [Conv(1x1)][Conv(1x1)][Conv(1x1)]   [Conv(1x1)]
                                  ^          ^          ^             ^
                                  |          |          |             |
            *          *         u_0        u_1        u_2           u_T         *
            ^          ^          ^          ^          ^             ^          ^
            |          |          |          |          |             |          |
        [Encoder]->[Encoder]->[Encoder]->[Decoder]->[Decoder]...->[Decoder]->[Decoder]
            ^          ^          ^
            |          |          |
           x_0        x_1        x_2

        '''
        # concatenate the outputs of encoder and decoder to get [z_0, ..., z_T]
        n_input_timesteps = self.x.shape[0]
        sequences = T.concatenate([self.encoder.outputs_all_layers, self.decoder.outputs_all_layers], axis=0)[n_input_timesteps-1:-1]
        n_output_timesteps = sequences.shape[0]

        # build function
        def step(u):
            self.conv_layer.input = u
            z = self.conv_layer.output
            return z

        # feed [u_0, ..., u_T] to step() and get [z_0, ..., z_T]
        rval, updates = theano.scan(
            step,
            sequences=sequences,
            n_steps=n_output_timesteps,
            name="{0}_scan".format(self.name)
        )
        self.rval = rval
        self.updates = updates

    @property
    def output(self):
        return self.outputs[-1]

    @property
    def outputs(self):
        return self.rval

    @property
    def params(self):
        return [
            self.encoder.params,
            self.decoder.params,
            self.conv_layer.params
        ]

    @params.setter
    def params(self, param_list):
        self.encoder.params = param_list[0]
        self.decoder.params = param_list[1]
        self.conv_layer.params = param_list[2]