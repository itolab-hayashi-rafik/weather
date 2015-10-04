# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import pdb, traceback

import time
import string
import math

import numpy
import theano
from theano import tensor as T

import dnn
from generator import ConstantGenerator, SinGenerator, RadarGenerator
import utils

class TestBed(object):
    def __init__(self, window_size=10, t_in=3, w=10, h=10, d=1, t_out=3, hidden_layers_sizes=[100,100]):
        '''
        初期化する
        :param window_size:
        :param t_in: DNN に入力する過去のデータの個数
        :param w: 各データの横幅
        :param h: 各データの高さ
        :param d: 各データのチャンネル数
        :param t_out: DNN から出力する未来のデータの個数
        :param hidden_layers_sizes: 中間層のユニット数
        :return:
        '''
        self.window_size = window_size
        self.t_in = t_in
        self.w = w
        self.h = h
        self.d = d
        self.t_out = t_out
        self.dataset = [ numpy.zeros((d,h,w), dtype=theano.config.floatX) for i in xrange(window_size) ]

        numpy_rng = numpy.random.RandomState(89677)
        # for each value n in hidden_layers_sizes, assume it as a filter of (1,d,sqrt(n),sqrt(n)), which means it has one sqrt(n)*sqrt(n) sized filter
        # filter_shapes = [(10,d,k,k) for k in hidden_layers_sizes]

        # self.model = dnn.SdAIndividual(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)
        # self.model = dnn.SdAFullyConnected(numpy_rng, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)

        # StackedLSTM を使う場合は hidden_layers_sizes が [...] + [n_ins] でないといけない.
        # self.model = dnn.StackedLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, hidden_layers_sizes=hidden_layers_sizes)

        # StackedConvLSTM では中間層の大きさは入力層と同じ(固定). ただしパラメータ数(フィルタの数, 大きさ)は自由に変えられる.
        # self.model = dnn.StackedConvLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, filter_shapes=filter_shapes)

        # EncoderDecoderLSTM を使う場合は hidden_layers_sizes が [n_ins] + [...] + [n_ins] でないといけない.
        self.model = dnn.EncoderDecoderLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, t_out=t_out, hidden_layers_sizes=hidden_layers_sizes)

        # EncoderDecoderConvLSTM では中間層の大きさは入力層と同じ(固定). ただしパラメータ数(フィルタの数, 大きさ)は自由に変えられる.
        # self.model = dnn.EncoderDecoderConvLSTM(numpy_rng, t_in=t_in, d=d, w=w, h=h, t_out=t_out, filter_shapes=filter_shapes)

        print('Building pretrain function...'),
        self.f_pretrain = self.model.build_pretrain_function()
        print('done')

        print('Building finetune function...'),
        self.f_grad_shared, self.f_update = self.model.build_finetune_function()
        print('done')

        print('Building predict function...'),
        self.f_predict = self.model.build_prediction_function()
        print('done')

    def supply(self, data):
        self.dataset.append(data)
        while self.window_size < len(self.dataset):
            self.dataset.pop(0)

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

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.t_in) for n in idx], :, : ,:]

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[range(n+self.t_in,n+self.t_in+self.t_out) for n in idx], :, :, :]

    def pretrain(self, epochs=15, learning_rate=0.1, batch_size=1):
        '''
        現在持っているデータセットで学習する
        :return:
        '''
        if self.f_pretrain is None:
            return numpy.inf

        dataset = numpy.asarray(self.dataset, dtype=theano.config.floatX)
        idx = range(self.window_size-self.t_in-self.t_out+1)
        numpy.random.shuffle(idx)
        cut = int(math.ceil(0.8*len(idx)))
        train_idx = idx[:cut]
        valid_idx = idx[cut:]
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
                x, mask, y = self.model.prepare_data(x, y)
                n_samples += x.shape[1]

                cost = self.f_pretrain(x, mask, y, learning_rate)

                avg_cost += cost / len(kf)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('NaN detected, cost={0}'.format(cost))
                    type, value, tb = sys.exc_info()
                    traceback.print_exc()
                    pdb.post_mortem(tb)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    # print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    pass

                if numpy.mod(uidx, validFreq) == 0:
                    #use_noise.set_value(0.) # TODO: implement dropout?
                    valid_costs = self.pred_error(dataset, valid_idx, batch_size)
                    valid_cost = numpy.mean(valid_costs)
                    v_costs.append(valid_cost)
                    history_errs.append(valid_cost)

                    if (uidx == 0 or
                                valid_cost <= numpy.array(history_errs).min()):
                        #best_p = unzip(tparams) # FIXME: saving parameters is not implemented here
                        bad_counter = 0

                    print(" (validtion) Train:{0}, Valid: {1}".format(cost, valid_cost))

                    if (len(history_errs) > patience and
                                valid_cost >= numpy.array(history_errs)[:-patience].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                costs.append(avg_cost)

            print("Epoch {0}/{1}: Seen {2} samples".format(eidx+1, epochs, n_samples))

            if estop:
                break

        return numpy.average(costs), numpy.average(v_costs), None

    def finetune(self, epochs=100, learning_rate=0.1, batch_size=1):
        '''
        現在持っているデータセットで学習する
        :return:
        '''
        print('finetune: learning_rate={0}'.format(learning_rate))
        dataset = numpy.asarray(self.dataset, dtype=theano.config.floatX)
        idx = range(self.window_size-self.t_in-self.t_out+1)
        numpy.random.shuffle(idx)
        cut = int(math.ceil(0.8*len(idx)))
        train_idx = idx[:cut]
        valid_idx = idx[cut:]
        n_train_batches = len(train_idx) / batch_size

        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless

        history_errs = []
        bad_counter = 0

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
                x, mask, y = self.model.prepare_data(x, y)
                n_samples += x.shape[1]

                cost = self.f_grad_shared(x, mask, y)
                self.f_update(learning_rate)

                avg_cost += cost / len(kf)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('NaN detected, cost={0}'.format(cost))
                    type, value, tb = sys.exc_info()
                    traceback.print_exc()
                    pdb.post_mortem(tb)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    # print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    pass

                if numpy.mod(uidx, validFreq) == 0:
                    #use_noise.set_value(0.) # TODO: implement dropout?
                    valid_costs = self.pred_error(dataset, valid_idx, batch_size)
                    valid_cost = numpy.mean(valid_costs)
                    v_costs.append(valid_cost)
                    history_errs.append(valid_cost)

                    if (uidx == 0 or
                                valid_cost <= numpy.array(history_errs).min()):
                        #best_p = unzip(tparams) # FIXME: saving parameters is not implemented here
                        bad_counter = 0

                    print(" (validtion) Train:{0}, Valid: {1}".format(cost, valid_cost))

                    if (len(history_errs) > patience and
                                valid_cost >= numpy.array(history_errs)[:-patience].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                costs.append(avg_cost)

            print("Epoch {0}/{1}: Seen {2} samples".format(eidx+1, epochs, n_samples))

            if estop:
                break

        return numpy.average(costs), numpy.average(v_costs), None

    def pred_error(self, dataset, idx, batch_size):
        # Get new shuffled index for the training set.
        kf = self.get_minibatches_idx(idx, batch_size, shuffle=True)

        # iteratively validate each minibatch
        costs = []
        for _, valid_index in kf:
            # Select the random examples for this minibatch
            y = self._make_output(dataset, valid_index)
            x = self._make_input(dataset, valid_index)

            x, mask, y = self.model.prepare_data(x, y)
            y_ = self.f_predict(x, mask)

            n_samples = y.shape[1]

            cost = -numpy.sum(y * numpy.log(y_) + (1.-y) * numpy.log(1.-y_)) / n_samples
            costs.append(cost)

        return costs

    def predict(self):
        '''
        現在のデータセットから将来のデータを予測する
        :return:
        '''
        idx = len(self.dataset)-self.t_in-1
        dataset = numpy.asarray(self.dataset, dtype=theano.config.floatX)
        x = self._make_input(dataset, [idx])
        x, mask, _ = self.model.prepare_data(x, None)
        y = self.f_predict(x, mask) # f_predict returns output of (n_timesteps, 1, n_feature_maps, height, width)
        y = y.swapaxes(0,1)[0]      # so we need to swap axes and get (n_timesteps, n_feature_maps, height, width)
        print('y.shape={0}'.format(y.shape))
        y = y.reshape((self.t_out, self.d, self.h, self.w))
        return y

    def save_params(self):
        params = self.model.params
        # TODO


if __name__ == '__main__':
    bed = TestBed()
    # gen = ConstantGenerator(w=bed.w, h=bed.h, d=bed.d)
    gen = SinGenerator(w=bed.w, h=bed.h, d=bed.d)
    # gen = RadarGenerator("../data/radar", w=bed.w, h=bed.h)

    # fill the window with data
    for i in xrange(bed.window_size):
        y = gen.next()
        bed.supply(y)

    for i,y in enumerate(gen):
        # predict
        y_pred = bed.predict()
        print("{0}: y={1}, y_pred={2}".format(i, y, y_pred))

        bed.supply(y)

        # if i % pretrain_step == 0 and 0 < self.pretrain_epochs:
        #     # pretrain
        #     avg_cost = self.bed.pretrain(self.pretrain_epochs, learning_rate=self.pretrain_lr, batch_size=self.pretrain_batch_size)
        #     print("   pretrain cost: {0}".format(avg_cost))
        #     pass

        # finetune
        avg_cost = bed.finetune()
        print(" finetune {0}, train cost: {1}".format(i,avg_cost))

        time.sleep(1)
