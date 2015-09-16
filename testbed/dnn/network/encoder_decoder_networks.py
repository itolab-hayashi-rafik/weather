# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T

from base import StandaloneNetwork, tensor5
from stacked_networks import StackedLSTMEncoder, StackedLSTMDecoder, StackedConvLSTMEncoder, StackedConvLSTMDecoder

class EncoderDecoderNetwork(StandaloneNetwork):
    '''
    Base implementation of Stacked Network
    '''
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 mask=None,
                 output=None
                 ):
        self.encoder = None
        self.decoder = None

        assert input is not None
        assert output is not None

        super(EncoderDecoderNetwork, self).__init__(numpy_rng, theano_rng, input, mask, output)

    def setup(self):
        '''
        Construct the stacked network
        :return:
        '''
        raise NotImplementedError

    @property
    def output(self):
        return self.decoder.output

    @property
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


class EncoderDecoderLSTM(EncoderDecoderNetwork):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 mask=None,
                 output=None,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
    ):
        '''

        :param numpy_rng:
        :param theano_rng:
        :param input: input tensor of shape (n_timesteps, n_samples, n_ins)
        :param mask: input mask of shape (n_timesteps, n_samples)
        :param output: output matrix of shape (n_samples, n_ins)
        :param n_ins:
        :param hidden_layers_sizes:
        :return:
        '''
        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes

        # Allocate symbolic variables for the data
        if input is None:
            # the input minibatch data is of shape (n_samples, n_ins)
            input = T.tensor3('x', dtype=theano.config.floatX)
        if mask is None:
            # the input minibatch mask is of shape (n_samples, n_ins)
            mask = T.matrix('mask', dtype=theano.config.floatX) # FIXME: not used
        if output is None:
            # the output minibatch data is of shape (n_samples, n_ins)
            output = T.matrix('y', dtype=theano.config.floatX)

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, theano_rng, input, mask, output)

    def setup(self):
        # Encoder network
        self.encoder = StackedLSTMEncoder(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask,
            output=self.y,
            n_ins=self.n_ins,
            hidden_layers_sizes=self.hidden_layers_sizes,
        )

        # Decoder network
        self.decoder = StackedLSTMDecoder(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask, # FIXME: is this ok?
            output=self.y,
            encoder=self.encoder,
            n_timesteps=1 # FIXME
        )


class EncoderDecoderConvLSTM(EncoderDecoderNetwork):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 mask=None,
                 output=None,
                 input_shape=(1,28,28),
                 filter_shapes=[(1,1,3,3)]
    ):
        '''

        :param numpy_rng:
        :param theano_rng:
        :param input: input 5D tensor of shape (n_timesteps, n_samples, n_feature_maps, height, width)
        :param mask: input mask of shape (n_timesteps, n_samples, n_feature_maps)
        :param output: output 4D tensor of shape (n_samples, n_feature_maps, height, width)
        :param input_shape: (num input feature maps, input height, input width)
        :param filter_shapes: [(number of filters, num input feature maps, filter height, filter width)]
        :return:
        '''
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes

        # Allocate symbolic variables for the data
        if input is None:
            # the input minibatch data is of shape (n_timestep, n_samples, n_feature_maps, height, width)
            input = tensor5('x', dtype=theano.config.floatX)
        if mask is None:
            # the input minibatch mask is of shape (n_timestep, n_samples, n_feature_maps)
            mask = T.tensor3('mask', dtype=theano.config.floatX) # FIXME: not used
        if output is None:
            # the output minibatch data is of shape (n_samples, n_feature_maps, height, width)
            output = T.tensor4('y', dtype=theano.config.floatX)

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, theano_rng, input, mask, output)

    def setup(self):
        # Encoder network
        self.encoder = StackedConvLSTMEncoder(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask,
            output=self.y,
            input_shape=self.input_shape,
            filter_shapes=self.filter_shapes
        )

        # Decoder network
        self.decoder = StackedConvLSTMDecoder(
            self.numpy_rng,
            theano_rng=self.theano_rng,
            input=self.x,
            mask=self.mask, # FIXME: is this ok?
            output=self.y,
            encoder=self.encoder,
            n_timesteps=2 # FIXME
        )

