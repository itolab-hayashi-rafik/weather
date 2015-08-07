import numpy
import theano
import theano.tensor as T

from model import Model
from network.stacked_lstm import StackedLSTM

class LSTMFullyConnected(Model):
    def __init__(self, numpy_rng, w=10, h=10, hidden_layers_sizes=[10]):
        self.w = w
        self.h = h
        self.n_inputs = w*h
        self.n_hidden_layers = len(hidden_layers_sizes)
        self.n_outputs = w*h

        print('LSTMFullyConnected: building the model...'),
        self.dnn = StackedLSTM(
            numpy_rng,
            n_ins=self.n_inputs,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=self.n_outputs
        )
        print('done')

        print('LSTMFullyConnected: building finetune function...'),
        self.f_grad_shared, self.f_update, self.f_validate = self.dnn.build_finetune_function()
        print('done')

        print('LSTMFullyConnected: building predict function...'),
        self.predict_fn = self.dnn.build_prediction_function()
        print('done')

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
        maxlen = numpy.max(lengths)

        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
        for idx, s in enumerate(xs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

        return x, x_mask, ys

    def get_minibatches_idx(self, idx, minibatch_size, shuffle=True):
        idx_list = numpy.asarray(idx, dtype="int32")

        if shuffle:
            numpy.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(len(idx) // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
            minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != idx[-1]):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def pretrain(self, dataset, epochs=100, learning_rate=0.1, batch_size=1):
        raise NotImplementedError("pretraining is not supported")

    def finetune(self, dataset, train_idx, valid_idx, epochs=100, learning_rate=0.1, batch_size=1):
        '''
        finetune the model using the dataset
        :param dataset: an array of ndarray of (d-by-h-by-w) dimention, whose size is bigger than n
        :return:
        '''
        n_train_batches = len(train_idx) / batch_size

        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless

        history_errs = []

        # bunch of configs
        dispFreq = 1
        validFreq = len(valid_idx) / batch_size

        # training phase
        uidx = 0  # the number of update done
        estop = False  # early stop
        for eidx in xrange(epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = self.get_minibatches_idx(train_idx, batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                #use_noise.set_value(1.) # TODO: implement dropout?

                # Select the random examples for this minibatch
                y = [dataset[1][t] for t in train_index]
                x = [dataset[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = self.prepare_data(x, y)
                n_samples += x.shape[1]

                cost = self.f_grad_shared(x, mask, y)
                self.f_update(learning_rate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if numpy.mod(uidx, validFreq) == 0:
                    #use_noise.set_value(0.) # TODO: implement dropout?
                    valid_costs = self.validate(dataset, valid_idx, batch_size)
                    valid_cost = numpy.mean(valid_costs)
                    history_errs.append(valid_cost)

                    if (uidx == 0 or
                                valid_cost <= numpy.array(history_errs)[:,
                                             0].min()):
                        #best_p = unzip(tparams) # FIXME: saving parameters is not implemented here
                        bad_counter = 0

                    print('Train ', cost, 'Valid ', valid_cost)

                    if (len(history_errs) > patience and
                                valid_cost >= numpy.array(history_errs)[:-patience,
                                             0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break
        return

    def validate(self, dataset, valid_idx, batch_size):
        n_validate_batches = len(valid_idx) / batch_size

        # Get new shuffled index for the training set.
        kf = self.get_minibatches_idx(valid_idx, batch_size, shuffle=True)

        # iteratively validate each minibatch
        costs = []
        for _, valid_index in kf:
            # Select the random examples for this minibatch
            y = [dataset[1][t] for t in valid_index]
            x = [dataset[0][t] for t in valid_index]

            x, mask, y = self.prepare_data(x, y)

            cost = self.f_validate(x, mask, y)
            costs.append(cost)

        return costs

    def train(self, dataset, train_idx, valid_idx, epochs=100, learning_rate=0.1, batch_size=1):
        return self.finetune(dataset, train_idx, valid_idx, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    def predict(self, dataset):
        x, mask = self.prepare_data(dataset, None) # FIXME: None should be an numpy array to avoid manipulation against None object
        return self.predict_fn(x, mask)