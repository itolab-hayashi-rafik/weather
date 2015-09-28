# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy
import theano
import theano.tensor as T
from theano.gof.utils import flatten

from testbed import dnn
from testbed.dnn import network
from testbed.dnn.network import tensor5

import optimizers as O

class EncoderDecoderConvLSTM(dnn.BaseModel):
    def __init__(self, numpy_rng, dataset_sizes, t_in=2, d=1, w=10, h=10, t_out=1, filter_shapes=[(1, 1, 3, 3)]):
        self.filter_shapes = filter_shapes
        self.x = tensor5('x', dtype=theano.config.floatX)
        self.mask = T.tensor3('mask', dtype=theano.config.floatX)
        self.y = tensor5('y', dtype=theano.config.floatX)

        dnn = network.EncoderDecoderConvLSTM(
            numpy_rng,
            input=self.x.dimshuffle(1,0,2,3,4),
            mask=self.mask.dimshuffle(1,0,2),
            output=self.y.dimshuffle(1,0,2,3,4),
            input_shape=(d,h,w),
            filter_shapes=filter_shapes,
            n_timesteps=t_out
        )

        self.train_set_x, self.train_set_y = self._shared(t_in, d, h, w, t_out, dataset_sizes[0])
        self.valid_set_x, self.valid_set_y = self._shared(t_in, d, h, w, t_out, dataset_sizes[1])
        self.test_set_x, self.test_set_y = self._shared(t_in, d, h, w, t_out, dataset_sizes[2])
        self.train_set_mask = theano.shared(numpy.ones((dataset_sizes[0], t_in, d), dtype=theano.config.floatX), borrow=True)
        self.valid_set_mask = theano.shared(numpy.ones((dataset_sizes[1], t_in, d), dtype=theano.config.floatX), borrow=True)
        self.test_set_mask = theano.shared(numpy.ones((dataset_sizes[2], t_in, d), dtype=theano.config.floatX), borrow=True)

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, dnn, t_in, d, w, h, t_out)

    def _shared(self, t_in, d, h, w, t_out, batch_size):
        shared_x = theano.shared(numpy.zeros((batch_size, t_in, d, h, w), dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(numpy.zeros((batch_size, t_out, d, h, w), dtype=theano.config.floatX), borrow=True)
        return shared_x, shared_y

    @property
    def params(self):
        params = super(EncoderDecoderConvLSTM, self).params
        params['filter_shapes'] = self.filter_shapes
        return params

    def build_finetune_function(self, optimizer=O.adadelta, batch_size=16, valid_batch_size=64):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        y = self.dnn.y
        y_ = self.dnn.output

        cost = T.mean((y - y_)**2)
        params = flatten(self.dnn.params)
        grads = T.grad(cost, params)

        f_grad_shared, f_update = optimizer(learning_rate, params, grads,
                                            self.x, self.mask, self.y,
                                            self.train_set_x, self.train_set_mask, self.train_set_y,
                                            index, batch_size,
                                            cost)

        f_valid = theano.function([index], cost,
                                  givens={
                                      self.x: self.valid_set_x[index * valid_batch_size: (index + 1) * valid_batch_size],
                                      self.mask: self.valid_set_mask[index * valid_batch_size: (index + 1) * valid_batch_size],
                                      self.y: self.valid_set_y[index * valid_batch_size: (index + 1) * valid_batch_size]
                                  },
                                  name='f_valid')

        f_test = theano.function([index], cost,
                                givens={
                                    self.x: self.test_set_x[index * valid_batch_size: (index + 1) * valid_batch_size],
                                    self.mask: self.test_set_mask[index * valid_batch_size: (index + 1) * valid_batch_size],
                                    self.y: self.test_set_y[index * valid_batch_size: (index + 1) * valid_batch_size]
                                },
                                name='f_test')

        return (f_grad_shared, f_update, f_valid, f_test)

    def set_datasets(self, datasets):
        train_set, valid_set, test_set = datasets
        self.train_set_x.set_value(train_set[0])
        self.train_set_y.set_value(train_set[1])
        self.valid_set_x.set_value(valid_set[0])
        self.valid_set_y.set_value(train_set[1])
        self.test_set_x.set_value(test_set[0])
        self.test_set_y.set_value(test_set[1])