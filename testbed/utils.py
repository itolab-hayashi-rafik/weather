# -*- coding: utf-8 -*-
import sys

import numpy
import theano
import theano.tensor as T

def ndarray(x):
    '''
    returns ndarray of x
    :param x:
    :return:
    '''
    if isinstance(x, numpy.ndarray):
        return x
    return numpy.asarray(x, dtype=theano.config.floatX)

def flatten(l):
    return list(flattened(l))

def flattened(i):
    for a in i:
        if hasattr(a, '__iter__'):
            for b in flatten(a):
                yield b
        else:
            yield a