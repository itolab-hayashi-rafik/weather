# -*- coding: utf-8 -*-
import os
import glob

import math
import numpy
import theano
import theano.tensor as T

import csv

class Generator(object):
    def __init__(self, w=10, h=10, d=1):
        self.w = w
        self.h = h
        self.d = d

        self.t = -1

    def __iter__(self):
        return self

    def next(self):
        self.t += 1

class SinGenerator(Generator):
    def __init__(self, w=10, h=10, d=1):
        super(SinGenerator, self).__init__(w=w, h=h, d=d)

    def next(self):
        super(SinGenerator, self).next()

        data = \
            [
                [
                    [
                        (1 + math.sin((self.t - i - j) / math.pi)) * 0.4
                        for i in xrange(self.w)
                    ] for j in xrange(self.h)
                ] for k in xrange(self.d)
            ]

        return numpy.asarray(data, dtype=theano.config.floatX)

class RadarGenerator(Generator):
    def __init__(self, dir, w=0, h=0):
        super(RadarGenerator, self).__init__(w=w, h=h, d=1)
        dir = os.path.join(os.path.dirname(__file__), dir)
        self.dir = dir

        cwd = os.getcwd()
        os.chdir(dir)
        self.files = glob.glob('*.csv')
        self.files.sort()
        os.chdir(cwd)

        self.i = -1

    def next(self):
        super(RadarGenerator, self).next()

        self.i = (self.i + 1) % len(self.files)

        data = []

        file = self.files[self.i]
        filepath = os.path.join(self.dir, file)
        with open(filepath) as f:
            reader = csv.reader(f)
            datetime    = next(reader)  # ヘッダーの読み飛ばし
            grid        = next(reader)  # ヘッダーの読み飛ばし
            header      = next(reader)  # ヘッダーの読み飛ばし
            location    = next(reader)  # ヘッダーの読み飛ばし
            range       = next(reader)  # ヘッダーの読み飛ばし

            n_rows, n_cols = map(lambda x: int(x), grid)

            for timeline in reader:
                chunk = []
                for row in xrange(n_rows):
                    line = next(reader)
                    chunk.append(line[:self.w] if 0 < self.w and self.w < n_cols else line)
                data.append(chunk[:self.h] if 0 < self.h and self.h < n_rows else chunk)

        return numpy.asarray(data, dtype=theano.config.floatX)