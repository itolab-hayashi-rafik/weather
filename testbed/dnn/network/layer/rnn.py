import numpy as np
import theano
import theano.tensor as T

mode = theano.Mode(linker='cvm')

class RNN(object):
    """    Recurrent neural network class

    Supported output types:
    real : linear output units, use mean-squared error
    binary : binary output units, use cross-entropy error
    softmax : single softmax out, use cross-entropy error

    """
    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh,
                 output_type='real'):

        self.input = input
        self.activation = activation
        self.output_type = output_type

        self.batch_size = T.iscalar()

        # theta is a vector of all trainable parameters
        # it represents the value of W, W_in, W_out, h0, bh, by
        theta_shape = n_hidden ** 2 + n_in * n_hidden + n_hidden * n_out + \
                      n_hidden + n_hidden + n_out
        self.theta = theano.shared(value=np.zeros(theta_shape,
                                                  dtype=theano.config.floatX))

        # Parameters are reshaped views of theta
        param_idx = 0  # pointer to somewhere along parameter vector

        # recurrent weights as a shared variable
        self.W = self.theta[param_idx:(param_idx + n_hidden ** 2)].reshape(
            (n_hidden, n_hidden))
        self.W.name = 'W'
        W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                              low=-0.01, high=0.01),
                            dtype=theano.config.floatX)
        param_idx += n_hidden ** 2

        # input to hidden layer weights
        self.W_in = self.theta[param_idx:(param_idx + n_in * \
                                          n_hidden)].reshape((n_in, n_hidden))
        self.W_in.name = 'W_in'
        W_in_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
                                                 low=-0.01, high=0.01),
                               dtype=theano.config.floatX)
        param_idx += n_in * n_hidden

        # hidden to output layer weights
        self.W_out = self.theta[param_idx:(param_idx + n_hidden * \
                                           n_out)].reshape((n_hidden, n_out))
        self.W_out.name = 'W_out'

        W_out_init = np.asarray(np.random.uniform(size=(n_hidden, n_out),
                                                  low=-0.01, high=0.01),
                                dtype=theano.config.floatX)
        param_idx += n_hidden * n_out

        self.h0 = self.theta[param_idx:(param_idx + n_hidden)]
        self.h0.name = 'h0'
        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        param_idx += n_hidden

        self.bh = self.theta[param_idx:(param_idx + n_hidden)]
        self.bh.name = 'bh'
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        param_idx += n_hidden

        self.by = self.theta[param_idx:(param_idx + n_out)]
        self.by.name = 'by'
        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        param_idx += n_out

        assert(param_idx == theta_shape)

        # for convenience
        self.params = [self.W, self.W_in, self.W_out, self.h0, self.bh,
                       self.by]

        # shortcut to norms (for monitoring)
        self.l2_norms = {}
        for param in self.params:
            self.l2_norms[param] = T.sqrt(T.sum(param ** 2))

        # initialize parameters
        # DEBUG_MODE gives division by zero error when we leave parameters
        # as zeros
        self.theta.set_value(np.concatenate([x.ravel() for x in
                                             (W_init, W_in_init, W_out_init, h0_init, bh_init, by_init)]))

        self.theta_update = theano.shared(
            value=np.zeros(theta_shape, dtype=theano.config.floatX))

        # recurrent function (using tanh activation function) and arbitrary output
        # activation function
        def step(x_t, h_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) + \
                                  T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        # Note the implementation of weight-sharing h0 across variable-size
        # batches using T.ones multiplying h0
        # Alternatively, T.alloc approach is more robust
        [self.h, self.y_pred], _ = theano.scan(step,
                                               sequences=self.input,
                                               outputs_info=[T.alloc(self.h0, self.input.shape[1],
                                                                     n_hidden), None])
        # outputs_info=[T.ones(shape=(self.input.shape[1],
        # self.h0.shape[0])) * self.h0, None])

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        self.L1 += abs(self.W_out.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        elif self.output_type == 'binary':
            # push through sigmoid
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)  # apply sigmoid
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
        elif self.output_type == 'softmax':
            # push through softmax, computing vector of class-membership
            # probabilities in symbolic form
            #
            # T.nnet.softmax will not operate on T.tensor3 types, only matrices
            # We take our n_steps x n_seq x n_classes output from the net
            # and reshape it into a (n_steps * n_seq) x n_classes matrix
            # apply softmax, then reshape back
            y_p = self.y_pred
            y_p_m = T.reshape(y_p, (y_p.shape[0] * y_p.shape[1], -1))
            y_p_s = T.nnet.softmax(y_p_m)
            self.p_y_given_x = T.reshape(y_p_s, y_p.shape)

            # compute prediction as class whose probability is maximal
            self.y_out = T.argmax(self.p_y_given_x, axis=-1)
            self.loss = lambda y: self.nll_multiclass(y)

        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        #
        # Theano's advanced indexing is limited
        # therefore we reshape our n_steps x n_seq x n_classes tensor3 of probs
        # to a (n_steps * n_seq) x n_classes matrix of probs
        # so that we can use advanced indexing (i.e. get the probs which
        # correspond to the true class)
        # the labels y also must be flattened when we do this to use the
        # advanced indexing
        p_y = self.p_y_given_x
        p_y_m = T.reshape(p_y, (p_y.shape[0] * p_y.shape[1], -1))
        y_f = y.flatten(ndim=1)
        return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                            ('y', y.type, 'y_out', self.y_out.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_out, y))
        else:
            raise NotImplementedError()