import numpy
import theano
import theano.tensor as T

from base import Layer
from rnn import RNN

class LSTM(RNN):
    """
    LSTM
    """
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.n_hidden = kwargs.get('n_hidden')


    def setup(self):
        # input gate for cells
        self.in_gate     = Layer(self.n_in + self.n_hidden, self.n_hidden, T.nnet.sigmoid, self.clip_gradients)
        # forget gate for cells
        self.forget_gate = Layer(self.n_in + self.n_hidden, self.n_hidden, T.nnet.sigmoid, self.clip_gradients)
        # input modulation for cells
        self.in_gate2    = Layer(self.n_in + self.n_hidden, self.n_hidden, self.activation, self.clip_gradients)
        # output modulation
        self.out_gate    = Layer(self.n_in + self.n_hidden, self.n_hidden, T.nnet.sigmoid, self.clip_gradients)

        # keep these layers organized
        self.internal_layers = [self.in_gate, self.forget_gate, self.in_gate2, self.out_gate]

        # store the memory cells in first n spots, and store the current
        # output in the next n spots:
        self.initial_hidden_state = theano.shared(
            (self.nrng.standard_normal((self.n_hidden*2,)) * 1. / self.n_hidden*2).astype(theano.config.floatX)
        )

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

        if self.h.ndim > 1:
            #previous memory cell values
            prev_c = h[:, :self.hidden_size]

            #previous activations of the hidden layer
            prev_h = h[:, self.hidden_size:]
        else:

            #previous memory cell values
            prev_c = h[:self.hidden_size]

            #previous activations of the hidden layer
            prev_h = h[self.hidden_size:]

        # input and previous hidden constitute the actual
        # input to the LSTM:
        if h.ndim > 1:
            obs = T.concatenate([x, prev_h], axis=1)
        else:
            obs = T.concatenate([x, prev_h])
        # TODO could we combine these 4 linear transformations for efficiency? (e.g., http://arxiv.org/pdf/1410.4615.pdf, page 5)
        # how much to add to the memory cells
        in_gate = self.in_gate.activate(obs)

        # how much to forget the current contents of the memory
        forget_gate = self.forget_gate.activate(obs)

        # modulate the input for the memory cells
        in_gate2 = self.in_gate2.activate(obs)

        # new memory cells
        next_c = forget_gate * prev_c + in_gate2 * in_gate

        # modulate the memory cells to create the new output
        out_gate = self.out_gate.activate(obs)

        # new hidden output
        next_h = out_gate * T.tanh(next_c)

        if h.ndim > 1:
            return T.concatenate([next_c, next_h], axis=1)
        else:
            return T.concatenate([next_c, next_h])