import math
import numpy
import theano
import theano.tensor as T

class Generator(object):
    def __init__(self, w=10, h=10, d=1):
        self.w = w
        self.h = h
        self.d = d

        self.t = 0

    def __iter__(self):
        return self

    def next(self):
        data =\
        [
            [
                [
                    math.sin((self.t - i - j) / 100.0 / math.pi)
                    for i in xrange(self.w)
                ] for j in xrange(self.h)
            ] for k in xrange(self.d)
        ]

        self.t += 1
        return numpy.asarray(data, dtype=theano.config.floatX)