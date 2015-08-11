import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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
            n_ins=784,
            hidden_layers_sizes=[500, 500],
            n_outs=10
    ):
        super(StackedLSTM, self).__init__()

        self.n_ins = n_ins
        self.n_outs = n_outs
        self.lstm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.tensor3('x', dtype=theano.config.floatX) # the input minibatch data
        self.mask = T.matrix('mask', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)  # the regression is presented as real values
        # end-snippet-1

        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # construct hidden layers
        self.lstm_layers = []
        for i, n_hidden in enumerate(hidden_layers_sizes):
            # determine input size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # build an LSTM layer
            layer = LSTM(n_in=input_size,
                         n_out=hidden_layers_sizes[i],
                         activation=T.tanh,
                         prefix="LSTM{}".format(i),
                         nrng=numpy_rng,
                         trng=theano_rng)
            self.lstm_layers.append(layer)

            self.params.extend(layer.params)

        # # We now need to add a logistic layer on top of the Stacked LSTM
        # self.linLayer = LinearRegression(
        #     input=self.lstm_layers[-1].output_h,
        #     n_in=hidden_layers_sizes[-1],
        #     n_out=n_outs,
        #     activation=T.tanh,
        #     prefix="linLayer",
        #     nrng=numpy_rng,
        #     trng=theano_rng,
        # )
        #
        # self.params.extend(self.linLayer.params)

        def step(m, x, *prev_states):
            x_ = x
            new_states = []
            for i, layer in enumerate(self.lstm_layers):
                h_, c_, _ = prev_states[i], prev_states[i+1], prev_states[i+2]
                layer_out = layer.step(m, x_, h_, c_)
                _, _, x_ = layer_out # hidden, c, output
                new_states += layer_out
            return new_states # FIXME: is this correct?

        rval, updates = theano.scan(
            step,
            sequences=[self.mask, self.x],
            n_steps=n_timesteps,
            outputs_info=(
                [
                    dict(initial=T.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, hidden_layers_sizes[i]), taps=[-1])
                        for j in xrange(3)
                            for i in xrange(self.n_layers)
                ]
            ) # FIXME: dim_proj --> self.n_ins --> hidden_layer_sizes[i]
        )

        self.y_pred = rval[-1]
        self.errors = (self.y_pred - self.y).norm(L=2) / n_timesteps
        self.finetune_cost = self.errors

    def build_finetune_function(self, optimizer=O.adadelta):
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        cost = self.finetune_cost
        grads = T.grad(cost, self.params)

        f_validate = theano.function([self.x, self.mask, self.y], cost)

        f_grad_shared, f_update = optimizer(learning_rate, self.params, grads,
                                            self.x, self.mask, self.y, cost)

        return (f_grad_shared, f_update, f_validate)

    def build_prediction_function(self):
        return theano.function(
            [self.x, self.mask],
            outputs=self.y_pred
        )