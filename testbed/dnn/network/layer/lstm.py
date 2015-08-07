import numpy
import theano
import theano.tensor as T

from base import Layer
from rnn import RNN

class LSTM(RNN):
    """
    LSTM
    see: https://github.com/JonathanRaiman/theano_lstm/blob/master/theano_lstm/__init__.py
    """
    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh, clip_gradients=False, **kwargs):
        self.n_hidden = n_hidden
        super(LSTM, self).__init__(input, n_in, n_out, activation=activation, clip_gradients=clip_gradients, **kwargs)

    def setup(self):
        # store the memory cells in first n spots, and store the current
        # output in the next n spots:
        initial_hidden_state = theano.shared(
            (self.nrng.standard_normal((self.n_hidden*2,)) * 1. / self.n_hidden*2).astype(theano.config.floatX)
        )
        if self.input.ndim > 1:
            self.h = T.repeat(initial_hidden_state, self.input.shape[0], axis=0)
        else:
            self.h = initial_hidden_state

        if self.h.ndim > 1:
            #previous memory cell values
            self.prev_c = self.h[:, :self.n_hidden]

            #previous activations of the hidden layer
            self.prev_h = self.h[:, self.n_hidden:]
        else:
            #previous memory cell values
            self.prev_c = self.h[:self.n_hidden]

            #previous activations of the hidden layer
            self.prev_h = self.h[self.n_hidden:]

        # input and previous hidden constitute the actual
        # input to the LSTM:
        if self.h.ndim > 1:
            obs = T.concatenate([self.input, self.prev_h], axis=1)
        else:
            obs = T.concatenate([self.input, self.prev_h])
        # TODO could we combine these 4 linear transformations for efficiency? (e.g., http://arxiv.org/pdf/1410.4615.pdf, page 5)

        # input gate for cells
        self.in_gate     = Layer(obs, self.n_in + self.n_hidden, self.n_hidden, T.nnet.sigmoid, self.clip_gradients)
        # forget gate for cells
        self.forget_gate = Layer(obs, self.n_in + self.n_hidden, self.n_hidden, T.nnet.sigmoid, self.clip_gradients)
        # input modulation for cells
        self.in_gate2    = Layer(obs, self.n_in + self.n_hidden, self.n_hidden, self.activation, self.clip_gradients)
        # output modulation
        self.out_gate    = Layer(obs, self.n_in + self.n_hidden, self.n_hidden, T.nnet.sigmoid, self.clip_gradients)

        # keep these layers organized
        self.internal_layers = [self.in_gate, self.forget_gate, self.in_gate2, self.out_gate]

    @property
    def params(self):
        return [param for layer in self.internal_layers for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    @property
    def output(self):
        """
        The hidden activation, h, of the network, along
        with the new values for the memory cells, c,
        Both are concatenated as follows:
        >      y = f( x, past )
        Or more visibly, with past = [prev_c, prev_h]
        > [c, h] = f( x, [prev_c, prev_h] )
        """
        # new memory cells
        next_c = self.forget_gate.output * self.prev_c + self.in_gate2.output * self.in_gate.output

        # new hidden output
        next_h = self.out_gate.output * T.tanh(next_c)

        if self.h.ndim > 1:
            return T.concatenate([next_c, next_h], axis=1)
        else:
            return T.concatenate([next_c, next_h])

    @property
    def output_y(self):
        return self.output[:self.n_out]

    @property
    def output_h(self):
        return self.output[self.n_out:]