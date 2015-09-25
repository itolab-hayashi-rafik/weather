# -*- coding: utf-8 -*-
__author__ = 'masayuki'

import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import datetime

import numpy
import theano

import dnn
import dnn.optimizers as O
from utils import ndarray


class Experiment(object):
    def __init__(self, model, save_to="out/states.npz"):
        self.model = model
        self.save_to = save_to
        self.best_params = None
        self.history_errs = []

        # check if the output directory exists and make directory if necessary
        outdir = os.path.dirname(saveto)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # build functions
        print('Building functions...'),
        self.f_grad_shared, self.f_update = model.build_finetune_function(optimizer=O.adadelta)
        self.f_predict = model.build_prediction_function()
        print('done')

    def train(self, datasets, learning_rate=0.1, max_epochs=100, batch_size=16, valid_batch_size=64, test_batch_size=64, patience=None, disp_freq=None, valid_freq=None, save_freq=None):
        '''
        :param datasets: a tuple of ((train_set_x, train_set_y), (valid_set_x, valid_set_y))
        :param max_epochs: maximum num of epochs
        :param batch_size: batch size
        '''
        train_set, valid_set, test_set = datasets

        kf_train = self.get_minibatches_idx(len(train_set[0]), batch_size)
        kf_valid = self.get_minibatches_idx(len(valid_set[0]), valid_batch_size)
        kf_test = self.get_minibatches_idx(len(test_set[0]), test_batch_size)

        # bunch of configs
        if patience is None:
            patience = 10
        if disp_freq is None:
            disp_freq = len(train_set[0]) / batch_size
        if valid_freq is None:
            valid_freq = len(train_set[0]) / batch_size
        if save_freq is None:
            save_freq = len(train_set[0]) / batch_size

        bad_counter = 0
        estop = False
        for itr, monitor in enumerate(self.itertrain(train_set, learning_rate, max_epochs, batch_size)):
            # unpack parameters
            eidx, uidx, cost = monitor

            if numpy.mod(itr, disp_freq) == 0:
                print('Epoch {0}, Update {1}, Cost: {2}'.format(eidx, uidx, cost))
                pass

            if saveto and numpy.mod(itr, save_freq) == 0:
                self.save()

            if numpy.mod(itr, valid_freq) == 0:
                #use_noise.set_value(0.) # TODO: implement dropout?
                train_err = self.pred_error(train_set, kf_train)
                valid_err = self.pred_error(valid_set, kf_valid)
                test_err = self.pred_error(test_set, kf_test)

                self.history_errs.append([train_err, valid_err, test_err])

                if (uidx == 0 or
                            valid_err <= numpy.array(self.history_errs).min()):
                    self.best_params = self.model.params
                    bad_counter = 0

                print(" (validtion) Train: {0}, Valid: {1}, Test: {2}".format(train_err, valid_err, test_err))

                if (len(self.history_errs) > patience and
                            valid_err >= numpy.array(self.history_errs)[:-patience].min()):
                    bad_counter += 1
                    if bad_counter > patience:
                        print('Early Stop!')
                        estop = True
                        break

        train_err = self.pred_error(train_set, kf_train)
        valid_err = self.pred_error(valid_set, kf_valid)
        test_err = self.pred_error(test_set, kf_test)

        return train_err, valid_err, test_err

    def itertrain(self, dataset, learning_rate, epochs, batch_size):
        uidx = 0  # the number of update done
        for eidx in xrange(epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = self.get_minibatches_idx(len(dataset[0]), batch_size, shuffle=True)

            for bidx, train_index in kf:
                uidx += 1
                #use_noise.set_value(1.) # TODO: implement dropout?

                # Select the random examples for this minibatch
                y = [dataset[1][t] for t in train_index]
                x = [dataset[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = self.model.prepare_data(x, y)
                n_samples += x.shape[1]

                cost = self.f_grad_shared(x, mask, y)
                self.f_update(learning_rate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('NaN detected, cost={0}'.format(cost))
                    raise Exception

                yield eidx, uidx, cost

            print("Epoch {0}/{1}: Seen {2} samples".format(eidx+1, epochs, n_samples))

    def validate(self, dataset, batch_size=64):
        kf_valid = self.get_minibatches_idx(len(dataset[0]), batch_size)
        return self.pred_error(dataset, kf_valid)

    def test(self, dataset, batch_size=64):
        kf_test = self.get_minibatches_idx(len(dataset[0]), batch_size)
        return self.pred_error(dataset, kf_test)

    def pred_error(self, dataset, iterator):
        valid_err = 0
        for _, valid_index in iterator:
            y = [dataset[1][t] for t in valid_index]
            x = [dataset[0][t] for t in valid_index]
            x, mask, y = self.model.prepare_data(x, y)

            y_ = self.f_predict(x, mask)
            err = numpy.mean((y - y_)**2)
            valid_err += err
        valid_err = 1. - numpy.asarray(valid_err, dtype=theano.config.floatX) / len(dataset[0])

        return valid_err

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """
        idx_list = numpy.arange(n, dtype="int32")

        if shuffle:
            numpy.random.shuffle(idx_list)

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

    def save(self):
        print('Saving...'),
        if self.best_params is not None:
            params = self.best_params
        else:
            params = unzip(self.model.params)
        numpy.savez(saveto, history_errs=self.history_errs, **params)
        # pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
        print('Done')


def unzip(params):
    return params # FIXME: need deepcopy

def load_mnist_dataset(file):
    nda = numpy.load(file)
    input_raw_data = nda['input_raw_data']
    clips = nda['clips']
    xs = [input_raw_data[clips[0,i,0]:clips[0,i,0]+clips[0,i,1]] for i in xrange(clips.shape[1])]
    ys = [input_raw_data[clips[1,i,0]:clips[1,i,0]+clips[1,i,1]] for i in xrange(clips.shape[1])]
    return (ndarray(xs), ndarray(ys))

def exp_moving_mnist(
        train_datasets=['../data/moving_mnist/out/moving-mnist-train.npz'],
        valid_datasets=['../data/moving_mnist/out/moving-mnist-valid.npz'],
        test_datasets=['../data/moving_mnist/out/moving-mnist-test.npz'],
        filter_shapes=[(1,1,3,3)],
        saveto='out/states.npz',
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=None,  # Compute the validation error after this number of update.
        validFreq=None,  # Compute the validation error after this number of update.
        saveFreq=None,  # Save the parameters after every saveFreq updates
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation set.
        test_batch_size=64,  # The batch size used for test set.
):
    '''
    make experiment on Moving MNIST dataset
    :param train_dataset:
    :param valid_dataset:
    :param test_dataset:
    :param filter_shapes:
    :param states_file:
    :return:
    '''

    # build model
    numpy_rng = numpy.random.RandomState(89677)
    model = dnn.EncoderDecoderConvLSTM(
        numpy_rng,
        t_in=10,
        d=1,
        w=64,
        h=64,
        t_out=10,
        filter_shapes=filter_shapes
    )

    exp = Experiment(model, saveto)

    learning_rate = 1e-3
    for i, (train_dataset, valid_dataset, test_dataset) in enumerate(zip(train_datasets, valid_datasets, test_datasets)):
        train_set = load_mnist_dataset(train_dataset)
        valid_set = load_mnist_dataset(valid_dataset)
        test_set = load_mnist_dataset(test_dataset)

        print("Iteration {0}".format(i))
        print("{0} train examples".format((len(train_set[0]))))
        print("{0} valid examples".format(len(valid_set[0])))
        print("{0} test examples".format(len(test_set[0])))

        datasets = (train_set, valid_set, test_set)

        train_err, valid_err, test_err = exp.train(datasets, learning_rate, max_epochs, batch_size,valid_batch_size,
                                                   test_batch_size, patience, dispFreq, validFreq, saveFreq)
        print("{0} Train finished. Train: {1}, Valid: {2}, Test: {3}".format(i, train_err, valid_err, test_err))



if __name__ == '__main__':
    argv = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argv) # 引数の個数

    if argc <= 1:
        print("Usage: $ python {0} [1|2|3|4|5]".format(argv[0]))
        quit()

    exp = int(argv[1])
    if exp == 1:
        filter_shapes = [(256,1,5,5)]
    elif exp == 2:
        filter_shapes = [(128,1,5,5),(128,128,5,5)]
    elif exp == 3:
        filter_shapes = [(128,1,5,5),(64,128,5,5),(64,64,5,5)]
    elif exp == 4:
        filter_shapes = [(128,1,9,9),(128,128,9,9)]
    elif exp == 5:
        filter_shapes = [(128,1,9,9),(64,128,9,9),(64,64,9,9)]
    elif exp == 6:
        filter_shapes = [(32,1,5,5)]
    else:
        raise NotImplementedError

    now = datetime.datetime.today()
    saveto = "out/states-{0}-{1}.npz".format(exp,now.strftime('%Y%m%d%I%M%S'))
    print('Save file: {0}'.format(saveto))

    print('begin experiment')
    exp_moving_mnist(filter_shapes=filter_shapes, saveto=saveto)
    print('finish experiment')