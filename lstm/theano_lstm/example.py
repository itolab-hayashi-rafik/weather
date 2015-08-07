'''
see: https://github.com/JonathanRaiman/theano_lstm/issues/8
'''
import numpy as np
import theano
import theano.tensor as T

from theano_lstm import (create_optimization_updates, Layer, LSTM, StackedCells)


def main():
    # Make a dataset where the network should learn whether the number 1 has been seen yet in the first column of
    # the input sequence.  This probably isn't really a good example use case for an LSTM, but it's simple.
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
                         layers=[hidden_layer_size] * num_hidden_layers,
                         activation=T.tanh,
                         celltype=LSTM)

    # Make the connections from the input to the first layer have linear activations.
    model.layers[0].in_gate2.activation = lambda x: x

    # Add an output layer to predict the labels for each time step.
    output_layer = Layer(hidden_layer_size, 1, T.nnet.sigmoid)
    model.layers.append(output_layer)

    def step(x, *prev_hiddens):
        activations = model.forward(x, prev_hiddens=prev_hiddens)
        return activations

    input_vec = T.vector('input_vec')
    input_mat = input_vec.dimshuffle((0, 'x'))

    result, _ = theano.scan(fn=step,
                            sequences=[input_mat],
                            outputs_info=([dict(initial=hidden_layer.initial_hidden_state, taps=[-1])
                                           for hidden_layer in model.layers[:-1]] +
                                          [dict(initial=T.zeros_like(model.layers[-1].bias_matrix), taps=[-1])]))

    target = T.vector('target')
    prediction = result[-1].T[0]

    cost = T.nnet.binary_crossentropy(prediction, target).mean()

    updates, _, _, _, _ = create_optimization_updates(cost, model.params)

    update_func = theano.function([input_vec, target], cost, updates=updates, allow_input_downcast=True)
    predict_func = theano.function([input_vec], prediction, allow_input_downcast=True)

    for cur_iter in range(num_iterations):
        for i, (example, label) in enumerate(zip(examples, labels)):
            c = update_func(example, label)
            if i % 100 == 0:
                print(""),
            print("")

    test_cases = [np.array([-1, 1, 0, 1, -2, 0, 1, 0, 2, 0], dtype=theano.config.floatX),
                  np.array([2, 2, 2, 0, 0, 0], dtype=theano.config.floatX),
                  np.array([-2, -2, -2, 0, 0, 0], dtype=theano.config.floatX),
                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0], dtype=theano.config.floatX),
                  np.array([2, 0, 0, 0, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0], dtype=theano.config.floatX),
                  np.array([2, 2, 2, 0, 0, 0, 2, 2, 2, 0], dtype=theano.config.floatX)]


    for example in test_cases:
        print("input\toutput")
        for x, pred in zip(example, predict_func(example)):
            print("{}\t{:.3f}".format(x,pred))
        print("")

if __name__ == "__main__":
    main()