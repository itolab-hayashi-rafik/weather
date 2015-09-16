# -*- coding: utf-8 -*-
import pdb, traceback, sys
from abc import ABCMeta, abstractmethod
import numpy
import theano

class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def pretrain(self, dataset, epochs=100, learning_rate=0.1, batch_size=1):
        return

    @abstractmethod
    def train(self, dataset, train_idx, valid_idx, epochs=100, learning_rate=0.1, batch_size=1):
        return

    @abstractmethod
    def predict(self, dataset):
        return


class BaseModel(Model):
    def __init__(self, numpy_rng, dnn, n=2, d=1, w=10, h=10):
        '''
        Initialize ConvLSTM Encoder-Decoder model
        :param numpy_rng:
        :param dnn: network use
        :param n:
        :param d:
        :param w:
        :param h:
        '''
        self.n = n
        self.d = d
        self.w = w
        self.h = h

        self.dnn = dnn

        print('Building finetune function...'),
        self.f_grad_shared, self.f_update, self.f_validate = self.dnn.build_finetune_function()
        print('done')

        print('Building predict function...'),
        self.f_predict = self.dnn.build_prediction_function()
        print('done')

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.n) for n in idx], :, : ,:]

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[n+self.n for n in idx], :, :, :]

    def prepare_data(self, xs, ys, maxlen=None):
        '''
        prepare data for inserting to RNN or LSTM
        see: /lstm/tutorial/imdb.py
        :param xs:
        :param ys:
        :param maxlen:
        :return: x, mask, y
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
        for idx, s in enumerate(xs):
            x[:lengths[idx], idx, :, :, :] = s
            x_mask[:lengths[idx], idx, :] = 1.
        y = numpy.asarray(ys, dtype=theano.config.floatX) if ys is not None else ys

        return x, x_mask, y

    def get_minibatches_idx(self, idx, minibatch_size, shuffle=True):
        idx_list = numpy.asarray(idx, dtype="int32")

        if shuffle:
            numpy.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(len(idx) // minibatch_size):
            minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != len(idx)):
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
        costs = []
        v_costs = []
        for eidx in xrange(epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = self.get_minibatches_idx(train_idx, batch_size, shuffle=True)

            avg_cost = 0
            for bidx, train_index in kf:
                uidx += 1
                #use_noise.set_value(1.) # TODO: implement dropout?

                # Select the random examples for this minibatch
                y = self._make_output(dataset, train_index)
                x = self._make_input(dataset, train_index)

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = self.prepare_data(x, y)
                n_samples += x.shape[1]

                cost = self.f_grad_shared(x, mask, y)
                self.f_update(learning_rate)

                avg_cost += cost / len(kf)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    type, value, tb = sys.exc_info()
                    traceback.print_exc()
                    pdb.post_mortem(tb)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    # print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    pass

                if numpy.mod(uidx, validFreq) == 0:
                    #use_noise.set_value(0.) # TODO: implement dropout?
                    valid_costs = self.validate(dataset, valid_idx, batch_size)
                    valid_cost = numpy.mean(valid_costs)
                    v_costs.append(valid_cost)
                    history_errs.append(valid_cost)

                    if (uidx == 0 or
                                valid_cost <= numpy.array(history_errs).min()):
                        #best_p = unzip(tparams) # FIXME: saving parameters is not implemented here
                        bad_counter = 0

                    print(" (validtion) Train:{}, Valid: {}".format(cost, valid_cost))

                    if (len(history_errs) > patience and
                                valid_cost >= numpy.array(history_errs)[:-patience].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                costs.append(avg_cost)

            print("Epoch {}/{}: Seen {} samples".format(eidx+1, epochs, n_samples))

            if estop:
                break

        return numpy.average(costs), numpy.average(v_costs), None

    def validate(self, dataset, valid_idx, batch_size):
        n_validate_batches = len(valid_idx) / batch_size

        # Get new shuffled index for the training set.
        kf = self.get_minibatches_idx(valid_idx, batch_size, shuffle=True)

        # iteratively validate each minibatch
        costs = []
        for _, valid_index in kf:
            # Select the random examples for this minibatch
            y = self._make_output(dataset, valid_index)
            x = self._make_input(dataset, valid_index)

            x, mask, y = self.prepare_data(x, y)

            cost = self.f_validate(x, mask, y)
            costs.append(cost)

        return costs

    def train(self, dataset, train_idx, valid_idx, epochs=100, learning_rate=0.1, batch_size=1):
        return self.finetune(dataset, train_idx, valid_idx, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    def predict(self, dataset):
        x = self._make_input(dataset, [len(dataset)-self.n])
        x, mask, _ = self.prepare_data(x, None) # FIXME: None should be an numpy array to avoid manipulation against None object
        y = self.f_predict(x, mask)
        y = y.reshape((self.d, self.h, self.w))
        return y