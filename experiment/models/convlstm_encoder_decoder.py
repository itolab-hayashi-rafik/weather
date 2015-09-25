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
    def __init__(self, numpy_rng, datasets, t_in=2, d=1, w=10, h=10, t_out=1, filter_shapes=[(1, 1, 3, 3)]):
        self.filter_shapes = filter_shapes
        self.x = tensor5('x', dtype=theano.config.floatX)
        self.mask = T.tensor3('mask', dtype=theano.config.floatX)
        self.y = tensor5('y', dtype=theano.config.floatX)

        dnn = network.EncoderDecoderConvLSTM(
            numpy_rng,
            input=self.x.dimshuffle(1,0,2,3,4),
            mask=self.mask.dimshuffle(1,0,2),
            output=self.y,
            input_shape=(d,h,w),
            filter_shapes=filter_shapes,
            n_timesteps=t_out
        )

        print('loading dataset...'),
        train_set, valid_set, test_set = datasets
        self.train_set_x, self.train_set_y = self._shared(train_set)
        self.valid_set_x, self.valid_set_y = self._shared(valid_set)
        self.test_set_x, self.test_set_y = self._shared(test_set)
        self.train_set_mask = numpy.ones((len(train_set[0]), t_in, d), dtype=theano.config.floatX)
        self.valid_set_mask = numpy.ones((len(valid_set[0]), t_in, d), dtype=theano.config.floatX)
        self.test_set_mask = numpy.ones((len(test_set[0]), t_in, d), dtype=theano.config.floatX)
        print('done.')

        super(EncoderDecoderConvLSTM, self).__init__(numpy_rng, dnn, t_in, d, w, h, t_out)

    def _shared(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
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

