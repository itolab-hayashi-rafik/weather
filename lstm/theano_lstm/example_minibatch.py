'''
see: https://github.com/JonathanRaiman/theano_lstm/issues/8
'''
import numpy as np
import theano
import theano.tensor as T

from theano_lstm import (create_optimization_updates, Layer, LSTM, StackedCells, masked_loss)

import random

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def main():
    # Make a dataset where the network should learn whether the number 1 has been seen yet in the first column of
    # the input sequence. This probably isn't really a good example use case for an LSTM, but it's simple.
    rng = np.random.RandomState(123456789)
    input_size = 1
    input_length = 10
    sample_size = 500
    num_iterations = 3
    examples = rng.choice([-2, -1, 0, 1, 2], (sample_size, input_length)).astype(theano.config.floatX)
    labels = np.array([[1 if np.sum(np.abs(x[:y + 1])) > 5 else 0 for y in range(len(x))]
                       for x in examples],
                      dtype=theano.config.floatX)

    hidden_layer_size = 10
    num_hidden_layers = 2

    model = StackedCells(input_size,
                         layers=[hidden_layer_size,hidden_layer_size],
                         activation=T.tanh,
                         celltype=LSTM)

    # Add an output layer to predict the labels for each time step.
    model.layers.append(Layer(hidden_layer_size, input_length, lambda x: T.nnet.sigmoid(x)[0]))

    def step(x, *prev_hiddens):
        activations = model.forward(x, prev_hiddens[:-1])
        return activations

    initial_obs = T.matrix('')
    #timesteps = T.iscalar('timesteps')

    result, _ = theano.scan(step,
                            sequences=initial_obs[:,:-1],
                            outputs_info=([dict(initial=hidden_layer.initial_hidden_state, taps=[-1])
                                           for hidden_layer in model.layers[:-1]] +
                                          [dict(initial=T.zeros_like(model.layers[-1].bias_matrix), taps=[-1])]))

    prediction = result[-1]

    predict_func = theano.function(initial_obs, prediction, allow_input_downcast=True)

    # get minibatches
    batches_idx = get_minibatches_idx(examples.shape[0], 5, shuffle=False)

    for cur_iter in range(num_iterations):
        for _, batch_idx in batches_idx:
            batch_example = examples[batch_idx,:]
            batch_label = labels[batch_idx,:]
            output_all = predict_func(batch_example)

if __name__ == "__main__":
    main()