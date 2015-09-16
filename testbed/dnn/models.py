# -*- coding: utf-8 -*-
import pdb, traceback, sys
import numpy
import theano
import theano.tensor as T

from base import BaseModel
import network

class EncoderDecoderConvLSTM(BaseModel):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, filter_shapes=[(1,1,3,3)]):
        '''

        :param numpy_rng:
        :param n:
        :param d:
        :param w:
        :param h:
        :param filter_shapes:
        :return:
        '''

        print('Building the model...'),
        dnn = network.EncoderDecoderConvLSTM(
            numpy_rng,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes
        )
        print('done')

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, dnn, n, d, w, h)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.n) for n in idx], :].reshape((len(idx), self.n, self.d, self.h, self.w))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[n+self.n for n in idx], :].reshape((len(idx), self.d, self.h, self.w))

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
        maxlen = numpy.max(lengths) # n_timesteps

        x = numpy.zeros((maxlen, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
        x_mask = numpy.zeros((maxlen, n_samples, self.d), dtype=theano.config.floatX)
        for idx, s in enumerate(xs):
            x[:lengths[idx], idx, :, :, :] = s
            x_mask[:lengths[idx], idx, :] = 1.

        return x, x_mask, ys


class EncoderDecoderLSTM(BaseModel):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, hidden_layers_sizes=[100]):
        '''

        :param numpy_rng:
        :param n:
        :param d:
        :param w:
        :param h:
        :param hidden_layers_sizes:
        :return:
        '''
        self.n_ins = d*h*w

        print('Building the model...'),
        dnn = network.EncoderDecoderLSTM(
            numpy_rng,
            n_ins=self.n_ins,
            hidden_layers_sizes=hidden_layers_sizes
        )
        print('done')

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, dnn, n, d, w, h)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.n) for n in idx], :].reshape((len(idx), self.n, self.n_ins))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[n+self.n for n in idx], :].reshape((len(idx), self.n_ins))

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
        maxlen = numpy.max(lengths) # n_timesteps

        x = numpy.zeros((maxlen, n_samples, self.n_ins), dtype=theano.config.floatX)
        x_mask = numpy.zeros((maxlen, n_samples), dtype=theano.config.floatX)
        for idx, s in enumerate(xs):
            x[:lengths[idx], idx, :] = s
            x_mask[:lengths[idx], idx] = 1.

        return x, x_mask, ys


class StackedConvLSTM(BaseModel):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, filter_shapes=[(1,1,3,3)]):
        '''

        :param numpy_rng:
        :param n:
        :param d:
        :param w:
        :param h:
        :param filter_shapes:
        :return:
        '''

        print('LSTMFullyConnected: building the model...'),
        dnn = network.StackedConvLSTM(
            numpy_rng,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes
        )
        print('done')

        super(StackedConvLSTM, self).__init__(numpy_rng, dnn, n, d, w, h)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.n) for n in idx], :].reshape((len(idx), self.n, self.d, self.h, self.w))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[n+self.n for n in idx], :].reshape((len(idx), self.d, self.h, self.w))

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
        maxlen = numpy.max(lengths) # n_timesteps

        x = numpy.zeros((maxlen, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
        x_mask = numpy.zeros((maxlen, n_samples, self.d), dtype=theano.config.floatX)
        for idx, s in enumerate(xs):
            x[:lengths[idx], idx, :, :, :] = s
            x_mask[:lengths[idx], idx, :] = 1.

        return x, x_mask, ys


class StackedLSTM(BaseModel):
    def __init__(self, numpy_rng, n=2, d=1, w=10, h=10, hidden_layers_sizes=[100]):
        '''

        :param numpy_rng:
        :param n:
        :param d:
        :param w:
        :param h:
        :param hidden_layers_sizes:
        :return:
        '''
        self.n_ins = d*h*w
        self.n_outs= d*h*w

        print('Building the model...'),
        dnn = network.StackedLSTM(
            numpy_rng,
            n_ins=self.n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
        )
        print('done')

        super(StackedLSTM, self).__init__(numpy_rng, dnn, n, d, w, h)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.n) for n in idx], :].reshape((len(idx), self.n, self.n_ins))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[n+self.n for n in idx], :].reshape((len(idx), self.n_outs))

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
        maxlen = numpy.max(lengths) # n_timesteps

        x = numpy.zeros((maxlen, n_samples, self.n_ins), dtype=theano.config.floatX)
        x_mask = numpy.zeros((maxlen, n_samples), dtype=theano.config.floatX)
        for idx, s in enumerate(xs):
            x[:lengths[idx], idx, :] = s
            x_mask[:lengths[idx], idx] = 1.

        return x, x_mask, ys