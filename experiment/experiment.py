# -*- coding: utf-8 -*-
__author__ = 'masayuki'

import os
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import datetime
import timeit
import numpy
import theano

from models.convlstm_encoder_decoder import EncoderDecoderConvLSTM
from testbed.utils import ndarray

def unzip(params):
    return params # FIXME: need deepcopy

def moving_mnist_load_dataset(train_dataset, valid_dataset, test_dataset):
    '''
    load datasets
    :param train_dataset:
    :param valid_dataset:
    :param test_dataset:
    :return:
    '''
    def load(file):
        nda = numpy.load(file)
        input_raw_data = nda['input_raw_data']
        clips = nda['clips']
        xs = [input_raw_data[clips[0,i,0]:clips[0,i,0]+clips[0,i,1]] for i in xrange(clips.shape[1])]
        ys = [input_raw_data[clips[1,i,0]:clips[1,i,0]+clips[1,i,1]] for i in xrange(clips.shape[1])]
        return (ndarray(xs), ndarray(ys))

    # load dataset
    train = load(train_dataset)
    valid = load(valid_dataset)
    test = load(test_dataset)

    # use only one part of the dataset to avoid MemoryError: alloc failed
    # train = (train[0][:1000], train[1][:1000])
    # valid = (valid[0][:200], valid[1][:200])
    # test = (test[0][:300], test[1][:300])

    return (train, valid, test)

def exp_moving_mnist(
        train_dataset='../data/moving_mnist/out/moving-mnist-train.npz',
        valid_dataset='../data/moving_mnist/out/moving-mnist-valid.npz',
        test_dataset='../data/moving_mnist/out/moving-mnist-test.npz',
        filter_shapes=[(1,1,3,3)],
        saveto='out/states.npz',
        patience=5000,  # Number of epoch to wait before early stop if no progress
        patience_increase = 2, # wait this much longer when a new best is found
        max_epochs=5000,  # The maximum number of epoch to run
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        learning_rate=1e-3
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

    # load dataset
    print('loading dataset...'),
    datasets = moving_mnist_load_dataset(train_dataset, valid_dataset, test_dataset)
    train_data, valid_data, test_data = datasets
    print('done')

    print("{0} train examples".format((len(train_data[0]))))
    print("{0} valid examples".format(len(valid_data[0])))
    print("{0} test examples".format(len(test_data[0])))

    # check if the output directory exists and make directory if necessary
    outdir = os.path.dirname(saveto)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # determine parameters
    d, h, w = train_data[0].shape[2], train_data[0].shape[3], train_data[0].shape[4]
    t_in, t_out = train_data[0].shape[1], train_data[1].shape[1]

    n_train_batches = len(train_data[0]) / batch_size
    n_valid_batches = len(valid_data[0]) / valid_batch_size
    n_test_batches = len(test_data[0]) / valid_batch_size

    # build model
    print('building model...'),
    numpy_rng = numpy.random.RandomState(89677)
    model = EncoderDecoderConvLSTM(numpy_rng, datasets, t_in=t_in, d=d, w=w, h=h, t_out=t_out, filter_shapes=filter_shapes)
    f_grad_shared, f_update, f_valid, f_test = model.build_finetune_function(batch_size=batch_size, valid_batch_size=valid_batch_size)
    print('done')

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < max_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = f_grad_shared(minibatch_index)
            f_update(learning_rate)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [f_valid(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [f_test(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    numpy.savez(saveto, **model.params)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


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
        filter_shapes = [(2,1,5,5),(2,2,9,9)]
    else:
        raise NotImplementedError

    now = datetime.datetime.today()
    saveto = "out/states-{0}-{1}.npz".format(exp,now.strftime('%Y%m%d%I%M%S'))
    print('Save file: {0}'.format(saveto))

    print('begin experiment')
    exp_moving_mnist(filter_shapes=filter_shapes, saveto=saveto)
    print('finish experiment')