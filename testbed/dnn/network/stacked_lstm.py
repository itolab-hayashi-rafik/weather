import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from base import Network
from layer.lstm import LSTM
from layer.linear_regression import LinearRegression

class StackedLSTM(Network):
    '''
    an implementation of Stacked LSTM
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

        self.lstm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.matrix('y')  # the regression is presented as real values
        # end-snippet-1

        # construct hidden layers
        self.lstm_layers = []
        for i, n_hidden in enumerate(hidden_layers_sizes):
            # determine input size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # determine input
            if i == 0:
                input = self.x
            else:
                input = self.lstm_layers[-1].output

            layer = LSTM(input=input,
                         n_in=input_size,
                         n_hidden=0,
                         n_out=hidden_layers_sizes[i],
                         nrng=numpy_rng)
            self.lstm_layers.append(layer)

            self.params.extend(layer.params)

        # We now need to add a logistic layer on top of the MLP
        self.linLayer = LinearRegression(
            rng=numpy_rng,
            input=self.lstm_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs,
            activation=T.tanh
        )

        self.params.extend(self.linLayer.params)

        self.finetune_cost = 0 # FIXME
        self.errors = 0 # FIXME
        self.y_pred = 0 # FIXME

    def build_finetune_function(self):
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]

        # TODO implement this

    def build_prediction_function(self):
        x = T.matrix('x')
        return theano.function(
            [x],
            outputs=self.y_pred,
            givens={
                self.x: x
            }
        )