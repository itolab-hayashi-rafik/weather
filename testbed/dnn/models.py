# -*- coding: utf-8 -*-
import pdb, traceback, sys
import numpy
import theano
import theano.tensor as T

from base import BaseModel
import network

class EncoderDecoderConvLSTM(BaseModel):
    def __init__(self, numpy_rng, theano_rng, t_in=2, d=1, w=10, h=10, t_out=1, filter_shapes=[(1,1,3,3)]):
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
        self.filter_shapes = filter_shapes

        dnn = network.EncoderDecoderConvLSTM2(
            numpy_rng=numpy_rng,
            theano_rng=theano_rng,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes,
            n_timesteps=t_out
        )

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, theano_rng, dnn, t_in, d, w, h, t_out)

    @property
    def params(self):
        params = BaseModel.params.fget(self)
        params['filter_shapes'] = self.filter_shapes
        return params

    @params.setter
    def params(self, param_list):
        BaseModel.params.fset(self, param_list)
        self.filter_shapes = param_list['filter_shapes']

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
    def __init__(self, numpy_rng, theano_rng, t_in=2, d=1, w=10, h=10, t_out=1, hidden_layers_sizes=[100]):
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
        self.hidden_layers_sizes = hidden_layers_sizes

        dnn = network.EncoderDecoderLSTM(
            numpy_rng=numpy_rng,
            theano_rng=theano_rng,
            n_ins=self.n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
            n_timesteps=t_out
        )

        super(EncoderDecoderLSTM, self).__init__(numpy_rng, theano_rng, dnn, t_in, d, w, h, t_out)

    @property
    def params(self):
        params = BaseModel.params.fget(self)
        params['hidden_layers_sizes'] = self.hidden_layers_sizes
        return params

    @BaseModel.params.setter
    def params(self, param_list):
        BaseModel.params.fset(self, param_list)
        self.hidden_layers_sizes = param_list['hidden_layers_sizes']

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
        for idx, xi in enumerate(xs):
            x[:lengths[idx], idx, :] = xi.reshape(xi.shape[0], self.n_ins)
            x_mask[:lengths[idx], idx] = 1.

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.n_ins), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :] = yi.reshape(yi.shape[0], self.n_ins)
        else:
            y = None

        return x, x_mask, y


class StackedConvLSTM(BaseModel):
    def __init__(self, numpy_rng, theano_rng, t_in=2, d=1, w=10, h=10, t_out=1, filter_shapes=[(1,1,3,3)]):
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
        self.filter_shapes = filter_shapes

        assert t_out == 1

        dnn = network.StackedConvLSTM(
            numpy_rng=numpy_rng,
            theano_rng=theano_rng,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes,
        )

        super(StackedConvLSTM, self).__init__(numpy_rng, theano_rng, dnn, t_in, d, w, h, t_out)

    @property
    def params(self):
        params = BaseModel.params.fget(self)
        params['filter_shapes'] = self.filter_shapes
        return params

    @BaseModel.params.setter
    def params(self, param_list):
        BaseModel.params.fset(self, param_list)
        self.filter_shapes = param_list['filter_shapes']

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


class StackedLSTM(BaseModel):
    def __init__(self, numpy_rng, theano_rng, t_in=2, d=1, w=10, h=10, t_out=1, hidden_layers_sizes=[100]):
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
        self.hidden_layers_sizes = hidden_layers_sizes

        assert t_out == 1

        dnn = network.StackedLSTM(
            numpy_rng=numpy_rng,
            theano_rng=theano_rng,
            n_ins=self.n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
        )

        super(StackedLSTM, self).__init__(numpy_rng, theano_rng, dnn, t_in, d, w, h, t_out)

    @property
    def params(self):
        params = BaseModel.params.fget(self)
        params['hidden_layers_sizes'] = self.hidden_layers_sizes
        return params

    @BaseModel.params.setter
    def params(self, param_list):
        BaseModel.params.fset(self, param_list)
        self.hidden_layers_sizes = param_list['hidden_layers_sizes']

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
        for idx, xi in enumerate(xs):
            x[:lengths[idx], idx, :] = xi.reshape(xi.shape[0], self.n_ins)
            x_mask[:lengths[idx], idx] = 1.

        if ys is not None:
            y = numpy.zeros((self.t_out, n_samples, self.n_ins), dtype=theano.config.floatX)
            for idx, yi in enumerate(ys):
                y[:, idx, :] = yi.reshape(yi.shape[0], self.n_ins)
        else:
            y = None

        return x, x_mask, y