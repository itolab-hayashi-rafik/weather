# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.configparser import config
from theano.tensor.basic import _multi
from theano.tensor.type import TensorType
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gof.utils import flatten

import optimizers as O

ctensor5 = TensorType('complex64', ((False,) * 5))
ztensor5 = TensorType('complex128', ((False,) * 5))
ftensor5 = TensorType('float32', ((False,) * 5))
dtensor5 = TensorType('float64', ((False,) * 5))
btensor5 = TensorType('int8', ((False,) * 5))
wtensor5 = TensorType('int16', ((False,) * 5))
itensor5 = TensorType('int32', ((False,) * 5))
ltensor5 = TensorType('int64', ((False,) * 5))


def tensor5(name=None, dtype=None):
    """Return a symbolic 4-D variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False, False))
    return type(name)
tensor5s, ftensor5s, dtensor5s, itensor5s, ltensor5s = _multi(
    tensor5, ftensor5, dtensor5, itensor5, ltensor5)


class Network(object):
    def __init__(self, numpy_rng, theano_rng=None):
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        # setup the network
        self.setup()

    def setup(self):
        pass

    @property
    def output(self):
        '''
        :return: the output of this network
        '''
        raise NotImplementedError

    @property
    def outputs(self):
        '''
        :return: list of output i.e. [output[0], output[1], ..., output[T]]
        '''
        raise NotImplementedError

    @property
    def params(self):
        '''
        :return: parameters in this network
        '''
        return []

    @params.setter
    def params(self, param_list):
        '''
        :param param_list: list of parameters
        '''
        pass


class StandaloneNetwork(Network):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 mask=None,
                 output=None
    ):
        self.x = input
        self.mask = mask
        self.y = output
        self.layers = []

        super(StandaloneNetwork, self).__init__(numpy_rng, theano_rng)

    @property
    def finetune_cost(self):
        '''
        :return: the cost of finetune
        '''
        return T.mean((self.output - self.y)**2)

    def build_finetune_function(self, optimizer=O.adadelta):
        '''
        build the finetune function
        :param optimizer: an optimizer to use
        :return:
        '''
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)

        cost = self.finetune_cost
        params = flatten(self.params)
        grads = T.grad(cost, params)

        f_validate = theano.function([self.x, self.mask, self.y], cost)

        f_grad_shared, f_update = optimizer(learning_rate, params, grads,
                                            self.x, self.mask, self.y, cost)

        return (f_grad_shared, f_update, f_validate)

    def build_prediction_function(self):
        return theano.function(
            [self.x, self.mask],
            outputs=self.output
        )