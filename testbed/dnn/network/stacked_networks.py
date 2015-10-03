# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gof.utils import flatten

from base import StandaloneNetwork, tensor5
from layer import Conv, LSTM, ConvLSTM

class StackedNetwork(StandaloneNetwork):
    '''
    Base implementation of Stacked Network
    '''
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="StackedNetwork",
                 input=None,
                 mask=None,
                 output=None,
                 is_rnn=False
    ):
        self.layers = []

        super(StackedNetwork, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn)

    def setup(self):
        '''
        Construct the stacked network
        :return:
        '''
        raise NotImplementedError

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def outputs(self):
        return self.layers[-1].outputs

    @property
    def params(self):
        return [[layer.params] for layer in self.layers]

    @params.setter
    def params(self, param_list):
        for layer, params in zip(self.layers, param_list):
            layer.params = params


class StackedLSTM(StackedNetwork):
    '''
    An implementation of Stacked LSTM
    '''
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="StackedLSTM",
                 input=None,
                 mask=None,
                 output=None,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 n_timesteps=None,
                 initial_hidden_states=None,
                 has_input=True
    ):
        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_layers = len(hidden_layers_sizes)
        self.initial_hidden_states = initial_hidden_states
        self.has_input = has_input

        # Allocate symbolic variables for the data
        if has_input:
            if input is None:
                # the input minibatch data is of shape (n_timesteps, n_samples, n_ins)
                input = T.tensor3('x', dtype=theano.config.floatX)
            if mask is None:
                # the input minibatch mask is of shape (n_samples, n_ins)
                mask = T.matrix('mask', dtype=theano.config.floatX) # FIXME: not used
        if output is None:
            # the output minibatch data is of shape (n_timesteps, n_samples, n_ins)
            output = T.tensor3('y', dtype=theano.config.floatX)

        self.n_timesteps  = input.shape[0] if has_input and n_timesteps is None else n_timesteps
        self.n_samples = input.shape[1] if has_input else None

        super(StackedLSTM, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # construct LSTM layers
        self.layers = []
        for i, n_hidden in enumerate(self.hidden_layers_sizes):
            # determine input size
            if i == 0:
                input_size = self.n_ins
                has_input = self.has_input
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                has_input = True

            # build an LSTM layer
            layer = LSTM(n_in=input_size,
                         n_out=self.hidden_layers_sizes[i],
                         has_input=has_input,
                         activation=T.tanh,
                         prefix="{0}_LSTM{1}".format(self.name,i),
                         nrng=self.numpy_rng,
                         trng=self.theano_rng)
            self.layers.append(layer)

        self.setup_scan()

    def setup_scan(self):
        # forward propagation
        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(m, x_, c_, h_)
                _, x_ = layer_out # c, h
                new_states += layer_out
            return new_states

        # set sequences and function
        if self.has_input:
            sequences = [self.mask, self.x]
            fn = lambda m, x, *prev_states: step(m, x, *prev_states)
        else:
            sequences = None
            fn = lambda *prev_states: step(None, None, *prev_states)

        # set initial hidden states
        if self.initial_hidden_states is not None:
            outputs_info = flatten(self.initial_hidden_states)
        else:
            outputs_info = []
            for layer in self.layers:
                outputs_info += layer.outputs_info(self.n_samples)

        # scan
        rval, updates = theano.scan(
            fn,
            sequences=sequences,
            n_steps=self.n_timesteps,
            outputs_info=outputs_info,
            name="{0}_scan".format(self.name)
        )
        self.rval = rval
        self.updates = updates

        # rval には n_timestamps 分の step() の戻り値 new_states が入っている
        # * rval[0]: (n_timesteps, n_samples, n_ins) の LSTM0_c
        # * rval[1]: (n_timesteps, n_samples, n_ins) の LSTM0_h
        # * rval[2]: (n_timesteps, n_samples, n_ins) の LSTM1_c
        # ...
        # * rval[-1]:(n_timesteps, n_samples, n_ins) の LSTMN_h

    @property
    def output(self):
        '''
        :return: the output of the last layer at the last time period
        '''
        return self.outputs[-1]

    @property
    def outputs(self):
        '''
        :return: the outputs of the last layer from time period 0 to T
        '''
        return self.rval[-1]

    @property
    def last_states(self):
        '''
        :return The states (c, h) of all ConvLSTMs at the last time interval T. This does not include
                the output of the encoder network, namely the output of Conv(1x1) layer
        '''
        return [
            [
                self.rval[2*i][-1],     # LSTM[i].c[T]
                self.rval[2*i+1][-1],   # LSTM[i].h[T]
            ] for i in xrange(self.n_layers)
        ]


class StackedLSTMEncoder(StackedNetwork):
    '''
    An implementation of Stacked LSTM Encoder
    '''
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="StackedLSTMEncoder",
                 input=None,
                 mask=None,
                 output=None,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 ):
        # in order to construct Encoder-Decoder network properly,
        # we need the output of the same size as the input
        assert n_ins == hidden_layers_sizes[0] and n_ins == hidden_layers_sizes[-1]

        self.n_ins = n_ins
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_layers = len(hidden_layers_sizes)

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

        super(StackedLSTMEncoder, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # construct LSTM layers
        self.layers = []
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
                         prefix="{0}_LSTM{1}".format(self.name,i),
                         nrng=self.numpy_rng,
                         trng=self.theano_rng)
            self.layers.append(layer)

        self.setup_scan()

    def setup_scan(self):
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        outputs_info = []
        for layer in self.layers:
            outputs_info += layer.outputs_info(n_samples)

        # feed forward calculation
        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.layers):
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
            name="{0}_scan".format(self.name)
        )
        self.rval = rval

        # rval には n_timestamps 分の step() の戻り値 new_states が入っている
        # * rval[0]: (n_timesteps, n_samples, n_ins) の LSTM0_c
        # * rval[1]: (n_timesteps, n_samples, n_ins) の LSTM0_h
        # * rval[2]: (n_timesteps, n_samples, n_ins) の LSTM1_c
        # ...
        # * rval[-1]:(n_timesteps, n_samples, n_ins) の LSTMN_h

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
        '''
        :return The states (c, h) of all ConvLSTMs at the last time interval T. This does not include
                the output of the encoder network, namely the output of Conv(1x1) layer
        '''
        return [
            [
                self.rval[2*i][-1],     # LSTM[i].c[T]
                self.rval[2*i+1][-1],   # LSTM[i].h[T]
            ] for i in xrange(self.n_layers)
        ]


class StackedLSTMDecoder(StackedNetwork):
    '''
    An implementation of Stacked LSTM Decoder
    '''

    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="StackedLSTMDecoder",
                 input=None,
                 mask=None,
                 output=None,
                 encoder=None,
                 n_timesteps=1
    ):
        assert encoder is not None
        assert n_timesteps is not None

        self.n_ins = encoder.n_ins
        self.hidden_layers_sizes = encoder.hidden_layers_sizes
        self.initial_hidden_states = encoder.last_states
        self.encoder = encoder
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

        super(StackedLSTMDecoder, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # construct LSTM layers
        self.layers = []
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
                         prefix="{0}_LSTM{1}".format(self.name,i),
                         nrng=self.numpy_rng,
                         trng=self.theano_rng)
            self.layers.append(layer)

        self.setup_scan()

    def setup_scan(self):
        n_timesteps = self.n_timesteps

        # set initial states of layers: flatten the given state list
        outputs_info  = flatten(self.initial_hidden_states)
        outputs_info += [self.encoder.outputs[-1]]

        # feed forward calculation
        def step(*prev_states):
            y_ = prev_states[-1]
            new_states = []
            for i, layer in enumerate(self.layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(1., y_, c_, h_)
                _, y_ = layer_out # c, h
                new_states += layer_out
            return new_states + [y_]

        rval, updates = theano.scan(
            step,
            n_steps=n_timesteps,
            outputs_info=outputs_info, # changed: dim_proj --> self.n_ins --> hidden_layer_sizes[i]
            name="{0}_scan".format(self.name)
        )
        self.rval = rval

        # rval には n_timestamps 分の step() の戻り値 new_states + [y_] が入っている
        # * rval[0]: (n_timesteps, n_samples, n_ins) の LSTM0_c
        # * rval[1]: (n_timesteps, n_samples, n_ins) の LSTM0_h
        # * rval[2]: (n_timesteps, n_samples, n_ins) の LSTM1_c
        # ...
        # * rval[-1]:(n_timesteps, n_samples, n_ins) の LSTMN_h

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



class StackedConvLSTM(StackedNetwork):
    '''
    an implementation of Stacked ConvLSTM
    see: https://github.com/JonathanRaiman/theano_lstm/blob/master/theano_lstm/__init__.py
    '''
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            name="StackedConvLSTM",
            input=None,
            mask=None,
            output=None,
            input_shape=(1,28,28),
            filter_shapes=[(1,1,3,3)],
            n_timesteps=None,
            initial_hidden_states=None,
            has_input=True
    ):
        '''
        Initialize StackedConvLSTM
        :param numpy_rng:
        :param theano_rng:

        :type input_shape: tuple or list of length 3
        :param input_shape: (num input feature maps, image height, image width)

        :type filter_shapes: list of "tuple or list of length 4"
        :param filter_shapes: [(number of filters, num input feature maps, filter height, filter width)]
        :return:
        '''
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.output_shape = (filter_shapes[-1][0], input_shape[1], input_shape[2])
        self.n_outs = numpy.prod(input_shape[1:])
        self.n_layers = len(filter_shapes)
        self.initial_hidden_states = initial_hidden_states
        self.has_input = has_input

        assert self.n_layers > 0

        # Allocate symbolic variables for the data
        if has_input:
            if input is None:
                # the input minibatch data is of shape (n_timesteps, n_samples, n_feature_maps, height, width)
                input = tensor5('x', dtype=theano.config.floatX)
            if mask is None:
                # the input minibatch mask is of shape (n_timesteps, n_samples, n_feature_maps)
                mask = T.tensor3('mask', dtype=theano.config.floatX) # FIXME: not used
        if output is None:
            # the output minibatch data is of shape (n_timesteps, n_samples, n_feature_maps, height, width)
            output = tensor5('y', dtype=theano.config.floatX)

        self.n_timesteps  = input.shape[0] if has_input and n_timesteps is None else n_timesteps
        self.n_samples = input.shape[1] if has_input else None

        super(StackedConvLSTM, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # construct LSTM layers
        for i, n_hidden in enumerate(self.filter_shapes):
            # determine input size
            if i == 0:
                input_shape = self.input_shape
                has_input = self.has_input
            else:
                input_shape = self.layers[-1].output_shape
                has_input = True

            # build an LSTM layer
            layer = ConvLSTM(input_shape=input_shape,
                             filter_shape=self.filter_shapes[i],
                             has_input=has_input,
                             activation=T.tanh,
                             prefix="{0}_ConvLSTM{1}".format(self.name,i),
                             nrng=self.numpy_rng,
                             trng=self.theano_rng)
            self.layers.append(layer)

        # setup feed forward formulation
        self.setup_scan()

    def setup_scan(self):
        # forward propagation
        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(m, x_, c_, h_)
                _, x_ = layer_out # c, h
                new_states += layer_out
            return new_states

        # set sequences and function
        if self.has_input:
            sequences = [self.mask, self.x]
            fn = lambda m, x, *prev_states: step(m, x, *prev_states)
        else:
            sequences = None
            fn = lambda *prev_states: step(None, None, *prev_states)

        # set initial states of layers
        if self.initial_hidden_states is not None:
            outputs_info = flatten(self.initial_hidden_states)
        else:
            outputs_info = []
            for layer in self.layers:
                outputs_info += layer.outputs_info(self.n_samples)

        # scan
        rval, updates = theano.scan(
            fn,
            sequences=sequences,
            n_steps=self.n_timesteps,
            outputs_info=outputs_info,
            name="{0}_scan".format(self.name)
        )
        self.rval = rval
        self.updates = updates

        # rval には n_timestamps 分の step() の戻り値 new_states が入っている
        # * rval[0]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM0_c
        # * rval[1]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM0_h
        # * rval[2]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM1_c
        # ...
        # * rval[-1]:(n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTMN_h

    @property
    def output(self):
        '''
        :return: the output of the last layer at the last time period
        '''
        return self.outputs[-1]

    @property
    def outputs(self):
        '''
        :return: the outputs of the last layer from time period 0 to T
        '''
        return self.rval[-1]

    @property
    def outputs_all_layers(self):
        '''
        :return: the outputs of all layers from time period 0 to T
        '''
        return T.concatenate(self.rval[1::2], axis=2)

    @property
    def last_states(self):
        '''
        :return The states (c, h) of all ConvLSTMs at the last time interval T. This does not include
                the output of the encoder network, namely the output of Conv(1x1) layer
        '''
        return [
            [
                self.rval[2*i][-1],     # ConvLSTM[i].c[T]
                self.rval[2*i+1][-1],   # ConvLSTM[i].h[T]
            ] for i in xrange(self.n_layers)
        ]

class StackedConvLSTMEncoder(StackedNetwork):
    '''
    An implementation of Stacked ConvLSTM Encoder
    '''
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            name="StackedConvLSTMEncoder",
            input=None,
            mask=None,
            output=None,
            input_shape=(1,28,28),
            filter_shapes=[(1,1,3,3)]
    ):
        '''
        Initialize StackedConvLSTM
        :param numpy_rng:
        :param theano_rng:

        :type input_shape: tuple or list of length 3
        :param input_shape: (num input feature maps, image height, image width)

        :type filter_shapes: list of "tuple or list of length 4"
        :param filter_shapes: [(number of filters, num input feature maps, filter height, filter width)]
        :return:
        '''
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.output_shape = input_shape
        self.n_ins = numpy.prod(self.input_shape)
        self.n_outs = numpy.prod(self.output_shape)
        self.n_layers = len(filter_shapes)

        # determine conv filter shape
        n_hiddens = sum([s[0] for s in filter_shapes]) # the number of total output feature maps (num of hidden states)
        self.conv_input_shape = (n_hiddens, input_shape[1], input_shape[2])
        self.conv_filter_shape = (input_shape[0], n_hiddens, 1, 1)

        assert self.n_layers > 0

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

        super(StackedConvLSTMEncoder, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # construct LSTM layers
        for i, n_hidden in enumerate(self.filter_shapes):
            # determine input size
            if i == 0:
                input_shape = self.input_shape
            else:
                input_shape = self.layers[-1].output_shape

            # build an LSTM layer
            layer = ConvLSTM(input_shape=input_shape,
                             filter_shape=self.filter_shapes[i],
                             activation=T.tanh,
                             prefix="{0}_ConvLSTM{1}".format(self.name,i),
                             nrng=self.numpy_rng,
                             trng=self.theano_rng)
            self.layers.append(layer)

        # construct Conv(1x1) layer
        self.conv_layer = Conv(
            None,
            self.conv_input_shape,
            self.conv_filter_shape,
            prefix="{0}_ConvLayer".format(self.name)
        )

        # setup feed forward formulation
        self.setup_scan()

    def setup_scan(self):
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # set initial states of layers
        outputs_info = []
        for layer in self.layers:
            outputs_info += layer.outputs_info(n_samples)
        # set initial output of conv layer
        outputs_info += [
            dict(initial=T.patternbroadcast(T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, *self.output_shape), [False, False, False, False]), taps=[-1])
        ]

        # feed forward calculation
        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(m, x_, c_, h_)
                _, x_ = layer_out # c, h
                new_states += layer_out

            # concatenate outputs of each ConvLSTM
            y_ = T.concatenate(new_states[1::2], axis=1) # concatenate h_ outputs of all layers
            self.conv_layer.input = y_ # a bit hacky way... should be fixed
            y_ = self.conv_layer.output

            # parameters to pass to next step are: hidden states and the output of the
            # decoder at this time interval (the input of the decoder at next time interval)
            return new_states + [y_]

        rval, updates = theano.scan(
            step,
            sequences=[self.mask, self.x],
            n_steps=n_timesteps,
            outputs_info=outputs_info,
            name="{0}_scan".format(self.name)
        )
        self.rval = rval

        # rval には n_timestamps 分の step() の戻り値 new_states + [y_] が入っている
        # * rval[0]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM0_c
        # * rval[1]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM0_h
        # * rval[2]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM1_c
        # ...
        # * rval[-1]:(n_timesteps, n_samples, n_output_feature_maps, height, width) の Conv(1x1) の出力

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
        '''
        :return The states (c, h) of all ConvLSTMs at the last time interval T. This does not include
                the output of the encoder network, namely the output of Conv(1x1) layer
        '''
        return [
            [
                self.rval[2*i][-1],     # ConvLSTM[i].c[T]
                self.rval[2*i+1][-1],   # ConvLSTM[i].h[T]
            ] for i in xrange(self.n_layers)
        ]

    @property
    def params(self):
        params  = StackedConvLSTM.params.fget(self)
        params += self.conv_layer.params
        return params

    @params.setter
    def params(self, param_list):
        StackedConvLSTM.params.fset(self, param_list[:-len(self.conv_layer.params)])
        self.conv_layer.params = param_list[len(self.conv_layer.params)-1:]


class StackedConvLSTMDecoder(StackedNetwork):
    '''
    An implementation of Stacked ConvLSTM Decoder
    '''
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 name="StackedConvLSTMDecoder",
                 input=None,
                 mask=None,
                 output=None,
                 encoder=None,
                 n_timesteps=1
    ):
        '''
        Initialize StackedConvLSTMDecoder
        :param numpy_rng:
        :param theano_rng:
        :return:
        '''
        assert encoder is not None
        input_shape = encoder.input_shape
        filter_shapes = encoder.filter_shapes
        output_shape = input_shape

        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.output_shape = output_shape
        self.initial_hidden_states = encoder.last_states
        self.encoder = encoder
        self.n_timesteps = n_timesteps

        self.n_ins = numpy.prod(self.input_shape)
        self.n_outs = numpy.prod(self.output_shape)
        self.n_layers = len(self.filter_shapes)

        # determine conv filter shape
        n_hiddens = sum([s[0] for s in self.filter_shapes]) # the number of total output feature maps (num of hidden states)
        self.conv_input_shape = (n_hiddens, input_shape[1], input_shape[2])
        self.conv_filter_shape = (input_shape[0], n_hiddens, 1, 1)

        assert self.n_layers > 0

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

        super(StackedConvLSTMDecoder, self).__init__(numpy_rng, theano_rng, name, input, mask, output, is_rnn=True)

    def setup(self):
        # construct LSTM layers
        for i, n_hidden in enumerate(self.filter_shapes):
            # determine input size
            if i == 0:
                input_shape = self.input_shape
            else:
                input_shape = self.layers[-1].output_shape

            # build an LSTM layer
            layer = ConvLSTM(input_shape=input_shape,
                             filter_shape=self.filter_shapes[i],
                             activation=T.tanh,
                             prefix="{0}_ConvLSTM{1}".format(self.name,i),
                             nrng=self.numpy_rng,
                             trng=self.theano_rng)
            self.layers.append(layer)

        # construct Conv(1x1) layer
        self.conv_layer = Conv(
            None,
            self.conv_input_shape,
            self.conv_filter_shape,
            prefix="{0}_ConvLayer".format(self.name)
        )

        # setup feed forward formulation
        self.setup_scan()

    def setup_scan(self):
        n_timesteps = self.n_timesteps

        # set initial states of layers: flatten the given state list
        outputs_info  = flatten(self.initial_hidden_states)
        # outputs_info += [self.x[-1]]
        # feed the last output of Encoder network as the first input of Decoder network
        outputs_info += [self.encoder.outputs[-1]]

        # feed forward calculation
        def step(*prev_states):
            y_ = prev_states[-1]

            # forward propagation
            new_states = []
            for i, layer in enumerate(self.layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(1., y_, c_, h_)
                _, y_ = layer_out # c, h
                new_states += layer_out

            # concatenate outputs of each ConvLSTM
            y_ = T.concatenate(new_states[1::2], axis=1) # concatenate h_ outputs of all layers
            self.conv_layer.input = y_ # a bit hacky way... should be fixed
            y_ = self.conv_layer.output # apply Conv(1x1) to outputs of all ConvLSTM

            # parameters to pass to next step are: hidden states and the output of the
            # decoder at this time interval (the input of the decoder at next time interval)
            return new_states + [y_]

        rval, updates = theano.scan(
            step,
            n_steps=n_timesteps,
            outputs_info=outputs_info,
            name="{0}_scan".format(self.name)
        )
        self.rval = rval

        # rval には n_timestamps 分の step() の戻り値 new_states + [y_] が入っている
        # * rval[0]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM0_c
        # * rval[1]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM0_h
        # * rval[2]: (n_timesteps, n_samples, n_output_feature_maps, height, width) の LSTM1_c
        # ...
        # * rval[-1]:(n_timesteps, n_samples, n_output_feature_maps, height, width) の Conv(1x1) の出力

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
    def params(self):
        params  = StackedConvLSTM.params.fget(self)
        params += self.conv_layer.params
        return params

    @params.setter
    def params(self, param_list):
        StackedConvLSTM.params.fset(self, param_list[:-len(self.conv_layer.params)])
        self.conv_layer.params = param_list[len(self.conv_layer.params)-1:]
