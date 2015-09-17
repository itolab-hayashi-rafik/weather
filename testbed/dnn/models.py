# -*- coding: utf-8 -*-
import pdb, traceback, sys
import numpy
import theano
import theano.tensor as T

from base import BaseModel
import network

class EncoderDecoderConvLSTM(BaseModel):
    def __init__(self, numpy_rng, t_in=2, d=1, w=10, h=10, t_out=1, filter_shapes=[(1,1,3,3)]):
        '''

        :param numpy_rng:
        :param t_in:
        :param d:
        :param w:
        :param h:
        :param t_out: num of output timesteps
        :param filter_shapes:
        :return:
        '''

        print('Building the model...'),
        dnn = network.EncoderDecoderConvLSTM(
            numpy_rng,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes,
            n_timesteps=t_out
        )
        print('done')

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, dnn, t_in, d, w, h, t_out)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.t_in) for n in idx], :].reshape((len(idx), self.t_in, self.d, self.h, self.w))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[range(n+self.t_in,n+self.t_in+self.t_out) for n in idx], :].reshape((len(idx), self.t_out, self.d, self.h, self.w))

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
        for idx, xi in enumerate(xs):
            x[:lengths[idx], idx, :, :, :] = xi
            x_mask[:lengths[idx], idx, :] = 1.

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :, :, :] = yi
        else:
            y = None

        return x, x_mask, y


class EncoderDecoderLSTM(BaseModel):
    def __init__(self, numpy_rng, t_in=2, d=1, w=10, h=10, t_out=1, hidden_layers_sizes=[100]):
        '''

        :param numpy_rng:
        :param t_in:
        :param d:
        :param w:
        :param h:
        :param t_out: num of output timesteps
        :param hidden_layers_sizes:
        :return:
        '''
        self.n_ins = d*h*w

        print('Building the model...'),
        dnn = network.EncoderDecoderLSTM(
            numpy_rng,
            n_ins=self.n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
            t_out=t_out
        )
        print('done')

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, dnn, t_in, d, w, h, t_out)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.t_in) for n in idx], :].reshape((len(idx), self.t_in, self.n_ins))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[range(n+self.t_in,n+self.t_in+self.t_out) for n in idx], :].reshape((len(idx), self.t_in, self.n_ins))

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

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.n_ins), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :] = yi
        else:
            y = None

        return x, x_mask, y


class StackedConvLSTM(BaseModel):
    def __init__(self, numpy_rng, t_in=2, d=1, w=10, h=10, t_out=1, filter_shapes=[(1,1,3,3)]):
        '''

        :param numpy_rng:
        :param t_in:
        :param d:
        :param w:
        :param h:
        :param t_out: num of output timesteps
        :param filter_shapes:
        :return:
        '''

        print('LSTMFullyConnected: building the model...'),
        dnn = network.StackedConvLSTM(
            numpy_rng,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes,
            t_out=t_out
        )
        print('done')

        super(StackedConvLSTM, self).__init__(numpy_rng, dnn, t_in, d, w, h, t_out)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.t_in) for n in idx], :].reshape((len(idx), self.t_in, self.d, self.h, self.w))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[range(n+self.t_in,n+self.t_in+self.t_out) for n in idx], :].reshape((len(idx), self.t_out, self.d, self.h, self.w))

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

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.d, self.h, self.w), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :, :, :] = yi
        else:
            y = None

        return x, x_mask, y


class StackedLSTM(BaseModel):
    def __init__(self, numpy_rng, t_in=2, d=1, w=10, h=10, t_out=1, hidden_layers_sizes=[100]):
        '''

        :param numpy_rng:
        :param t_in:
        :param d:
        :param w:
        :param h:
        :param t_out: num of output timesteps
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
            t_out=t_out
        )
        print('done')

        super(StackedLSTM, self).__init__(numpy_rng, dnn, t_in, d, w, h, t_out)

    def _make_input(self, dataset, idx):
        '''
        (i,j) の SdA に対する入力ベクトルを ndata から作る
        :param ndata: an array of ndarray of (d-by-h-by-w) dimention, whose size is n
        :return:
        '''
        return dataset[[range(n,n+self.t_in) for n in idx], :].reshape((len(idx), self.t_in, self.n_ins))

    def _make_output(self, dataset, idx):
        '''
        (i,j) の SdA に対する出力ベクトルをつくる
        :param data:
        :return:
        '''
        return dataset[[range(n+self.t_in,n+self.t_in+self.t_out) for n in idx], :].reshape((len(idx), self.t_out, self.n_outs))

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

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.n_ins), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :] = yi
        else:
            y = None

        return x, x_mask, y