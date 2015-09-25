# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy
import theano
import theano.tensor as T
from theano.gof.utils import flatten

from testbed import dnn
from testbed.dnn.network.base import tensor5
from testbed.dnn.network.layer import ConvLSTM, Conv
import testbed.dnn.optimizers as O

class ConvLSTMEncoderDecoder:
    def __init__(self, nrng, t_in, d, w, h, t_out, filter_shapes=[(1,1,3,3)]):
        # Allocate symbolic variables for the data
        self.x = tensor5('x', dtype=theano.config.floatX)
        self.mask = T.tensor3('mask', dtype=theano.config.floatX) # FIXME: not used
        self.y = tensor5('y', dtype=theano.config.floatX)

        self.encode_layers = []
        self.decode_layers = []
        self.params = []

        self.t_in = t_in
        self.d = d
        self.w = w
        self.h = h
        self.t_out = t_out

        self.input_shape = (d, h, w)
        n_samples = self.x.shape[1]
        n_hiddens = sum([s[0] for s in filter_shapes])

        ##
        ### Encoder
        ##

        # construct encoder
        for i, n_hidden in enumerate(filter_shapes):
            # determine input size
            if i == 0:
                input_shape = self.input_shape
            else:
                input_shape = self.encode_layers[-1].output_shape

            # build an LSTM layer
            layer = ConvLSTM(input_shape=input_shape,
                             filter_shape=filter_shapes[i],
                             activation=T.tanh,
                             prefix="Encoder_ConvLSTM{0}".format(i),
                             nrng=nrng)
            self.encode_layers.append(layer)
            self.params.extend(layer.params)

        # set initial states of layers
        outputs_info = []
        for layer in self.encode_layers:
            outputs_info += layer.outputs_info(n_samples)

        # feed forward calculation
        def encode_step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.encode_layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(m, x_, c_, h_)
                _, x_ = layer_out # hidden, c
                new_states += layer_out
            return new_states

        # scan
        rval, updates = theano.scan(
            encode_step,
            sequences=[self.mask, self.x],
            n_steps=t_in,
            outputs_info=outputs_info, # changed: dim_proj --> self.n_ins --> hidden_layer_sizes[i]
            name="Encoder_scan"
        )

        # extract hidden_states
        hidden_states = \
        [
            [
                rval[2*i][-1],     # ConvLSTM[i].c[T]
                rval[2*i+1][-1],   # ConvLSTM[i].h[T]
            ] for i in xrange(len(self.encode_layers))
        ]


        ##
        ### Decoder
        ##

        # construct decoder
        for i, n_hidden in enumerate(filter_shapes):
            # determine input size
            if i == 0:
                input_shape = self.input_shape
            else:
                input_shape = self.decode_layers[-1].output_shape

            # build an LSTM layer
            layer = ConvLSTM(input_shape=input_shape,
                             filter_shape=filter_shapes[i],
                             activation=T.tanh,
                             prefix="Decoder_ConvLSTM{0}".format(i),
                             nrng=nrng)
            self.decode_layers.append(layer)
            self.params.extend(layer.params)

        conv_layer = Conv(
            None,
            input_shape=(n_hiddens, self.input_shape[1], self.input_shape[2]),
            filter_shape=(self.input_shape[0], n_hiddens, 1, 1),
            prefix="Decoder_Conv"
        )
        self.params.extend(conv_layer.params)

        # set initial states of layers: flatten the given state list
        outputs_info  = flatten(hidden_states)
        outputs_info += [self.x[-1]]

        # feed forward calculation
        def decode_step(*prev_states):
            y_ = prev_states[-1]

            # forward propagation
            new_states = []
            for i, layer in enumerate(self.decode_layers):
                c_, h_ = prev_states[2*i], prev_states[2*i+1]
                layer_out = layer.step(1., y_, c_, h_)
                _, y_ = layer_out # c, h
                new_states += layer_out

            # concatenate outputs of each ConvLSTM
            y_ = T.concatenate(new_states[1::2], axis=1) # concatenate h_ outputs of all layers
            conv_layer.input = y_ # a bit hacky way... should be fixed
            y_ = conv_layer.output

            # parameters to pass to next step are: hidden states and the output of the
            # decoder at this time interval (the input of the decoder at next time interval)
            return new_states + [y_]

        # scan
        rval, updates = theano.scan(
            decode_step,
            n_steps=t_out,
            outputs_info=outputs_info,
            name="Decoder_scan"
        )

        self.outputs = rval[-1]

    def build_finetune_function(self, optimizer=O.adadelta):
        '''
        build the finetune function
        :param optimizer: an optimizer to use
        :return:
        '''
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        y = self.y
        y_ = self.outputs

        cost = T.mean((y - y_)**2)
        params = flatten(self.params)
        grads = T.grad(cost, params)

        f_grad_shared, f_update = optimizer(learning_rate, params, grads,
                                            self.x, self.mask, self.y, cost)

        return (f_grad_shared, f_update)

    def build_prediction_function(self):
        return theano.function([self.x, self.mask], outputs=self.outputs)

    def prepare_data(self, xs, ys, maxlen=None):
        '''
        prepare data for inserting to RNN or LSTM
        see: /lstm/tutorial/imdb.py
        :param xs:
        :param ys:
        :param maxlen:
        :return:
        '''
        lengths = [len(s) for s in xs]

        if maxlen is not None:
            new_seqs = []
            new_labels = []
            new_lengths = []
            for l, s, y in zip(lengths, xs, ys):
                if l < maxlen:
                    new_seqs.append(s)
                    new_labels.append(y)
                    new_lengths.append(l)
            lengths = new_lengths
            ys = new_labels
            xs = new_seqs

            if len(lengths) < 1:
                return None, None, None

        n_samples = len(xs)
        maxlen = numpy.max(lengths) # n_timesteps

        x = numpy.zeros((maxlen, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
        x_mask = numpy.zeros((maxlen, n_samples, self.d), dtype=theano.config.floatX)
        for idx, xi in enumerate(xs):
            x[:lengths[idx], idx, :, :, :] = xi
            x_mask[:lengths[idx], idx, :] = 1.

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :, :, :] = yi
        else:
            y = None

        return x, x_mask, y