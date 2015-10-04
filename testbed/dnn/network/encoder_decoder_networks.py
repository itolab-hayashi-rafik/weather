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
                 target=None,
                 is_rnn=False
                 ):
        self.encoder = None
        self.decoder = None

        assert input is not None
        assert target is not None

        super(EncoderDecoderNetwork, self).__init__(numpy_rng, theano_rng, name, input, mask, target, is_rnn)

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
                 target=None,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 n_timesteps=1
    ):
        '''

        :param numpy_rng:
        :param theano_rng:
        :param input: input tensor of shape (n_timesteps, n_samples, n_ins)
        :param mask: input mask of shape (n_timesteps, n_samples)
        :param target: output tensor of shape (n_timesteps, n_samples, n_ins)
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
        if target is None:
            # the output minibatch data is of shape (n_timesteps, n_samples, n_ins)
            target = T.tensor3('y', dtype=theano.config.floatX)

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, theano_rng, name, input, mask, target, is_rnn=True)

    def setup(self):
        # Encoder network
        self.encoder = StackedLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedLSTMEncoder",
            input=self.x,
            mask=self.mask,
            target=self.y,
            n_ins=self.n_ins,
            hidden_layers_sizes=self.hidden_layers_sizes,
        )

        # Decoder network
        self.decoder = StackedLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedLSTMDecoder",
            target=self.y,
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
                 target=None,
                 input_shape=(1,28,28),
                 filter_shapes=[(1,1,3,3)],
                 n_timesteps=1
    ):
        '''

        :param numpy_rng:
        :param theano_rng:
        :param input: input 5D tensor of shape (n_timesteps, n_samples, n_feature_maps, height, width)
        :param mask: input mask of shape (n_timesteps, n_samples, n_feature_maps)
        :param target: target 5D tensor of shape (n_timestamps, n_samples, n_feature_maps, height, width)
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
        if target is None:
            # the target minibatch data is of shape (n_timesteps, n_samples, n_feature_maps, height, width)
            target = tensor5('y', dtype=theano.config.floatX)

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, theano_rng, name, input, mask, target, is_rnn=True)

    def setup(self):
        # Encoder network
        self.encoder = StackedConvLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedConvLSTMEncoder",
            input=self.x,
            mask=self.mask,
            target=self.y,
            input_shape=self.input_shape,
            filter_shapes=self.filter_shapes
        )

        # Decoder network
        self.decoder = StackedConvLSTM(
            numpy_rng=self.numpy_rng,
            theano_rng=self.theano_rng,
            name="StackedConvLSTMDecoder",
            target=self.y,
            input_shape=self.input_shape,
            filter_shapes=self.filter_shapes,
            n_timesteps=self.n_timesteps,
            initial_hidden_states=self.encoder.last_states,
            has_input=False
        )

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
        conc = T.concatenate([self.encoder.outputs_all_layers[-1:], self.decoder.outputs_all_layers[:-1]], axis=0)

        # Here, the concatenated 5D tensor is of shape (n_timesteps, n_samples, n_hidden_feature_maps, height, width).
        # In order to input this to the Conv(1x1) network, we reshape this to 4D tensor of shape
        # (n_timesteps*n_samples, n_hidden_feature_maps, height, width)
        conv_input = conc.reshape((conc.shape[0]*conc.shape[1], conc.shape[2], conc.shape[3], conc.shape[4]))

        # Conv(1x1) layer
        self.conv_layer = Conv(
            conv_input,
            self.conv_input_shape,
            self.conv_filter_shape,
            prefix="{0}_ConvLayer".format(self.name)
        )
        conv_output = self.conv_layer.output

        # the output of Conv(1x1) layer is of shape (n_timesteps*n_samples, n_input_feature_maps, height, width),
        # so we reshape it to (n_timesteps, n_samples, n_input_feature_maps, height, width)
        self.net_output = conv_output.reshape((conc.shape[0], conc.shape[1]) + self.conv_layer.output_shape)

    @property
    def output(self):
        return self.outputs[-1]

    @property
    def outputs(self):
        return self.net_output

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