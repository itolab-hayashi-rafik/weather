# -*- coding: utf-8 -*-
import os
import glob

import math
import numpy
import theano
import theano.tensor as T

import csv
from PIL import Image

# Earth Radius (unit: meter)
R = 6371000

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

class ConstantGenerator(Generator):
    def __init__(self, w=10, h=10, d=1, value=0):
        self.value = value
        super(ConstantGenerator, self).__init__(w, h, d)

    def next(self):
        super(ConstantGenerator, self).next()

        data = \
            [
                [
                    [
                        self.value
                        for i in xrange(self.w)
                    ] for j in xrange(self.h)
                ] for k in xrange(self.d)
            ]

        return numpy.asarray(data, dtype=theano.config.floatX)


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
    def __init__(self, dir, w=0, h=0, offset=(0,0,0)):
        '''

        :param dir:
        :param w:
        :param h:
        :param offset: offsets of (x, y, timestep)
        :return:
        '''
        super(RadarGenerator, self).__init__(w=w, h=h, d=1)
        dir = os.path.join(os.path.dirname(__file__), dir)
        self.dir = dir
        self.offset = offset

        cwd = os.getcwd()
        os.chdir(dir)
        self.files = glob.glob('*.csv')
        self.files.sort()
        os.chdir(cwd)

        self.i = -1
        self.i += offset[2]

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

            w = self.w if 0 < self.w else n_cols
            h = self.h if 0 < self.h else n_rows
            w = w - self.offset[0] if n_cols < self.offset[0] + w else w
            h = h - self.offset[1] if n_rows < self.offset[1] + h else h

            for timeline in reader:
                chunk = []
                for row in xrange(n_rows):
                    line = next(reader)
                    chunk.append(line[self.offset[0]:self.offset[0]+w])
                data.append(chunk[self.offset[1]:self.offset[1]+h])

        return numpy.asarray(data, dtype=theano.config.floatX) / 100.0

class SatelliteGenerator(Generator):
    def __init__(self, dir, w=10, h=10, offset=(0,0,0), meshsize=(45,30), basepos=(491400,124200), imgbasepos=(), imgbaselng=135, imgscale=1.0):
        '''

        :param dir:
        :param w:
        :param h:
        :param offset: offsets of (x, y, timestep)
        :param meshsize: the size of each cell in the grid (unit: sec)
        :param basepos: the lat long position of the northwest (unit: sec)
        :param imgbasepos:
        :param imgbaselng:
        :param imgscale:
        :param
        :return:
        '''
        super(SatelliteGenerator, self).__init__(w, h, 1)
        dir = os.path.join(os.path.dirname(__file__), dir)
        self.dir = dir
        self.offset = offset
        self.meshsize = meshsize
        self.basepos = basepos
        self.imgbasepos = imgbasepos
        self.imgbaselng = imgbaselng
        self.imgscale = imgscale

        cwd = os.getcwd()
        os.chdir(dir)
        self.files = glob.glob('*.csv')
        self.files.sort()
        os.chdir(cwd)

        self.i = -1
        self.i += offset[2]

    def next(self):
        super(SatelliteGenerator, self).next()

        self.i = (self.i + 1) % len(self.files)
        file = self.files[self.i]
        filepath = os.path.join(self.dir, file)
        img = Image.open(filepath)

        def getval(lat, lng, d):
            phi = math.radians(lat)
            lmd = math.radians(lng - self.imgbaselng)
            p = 2. * R * math.tan((math.pi/2. - phi) / 2.)
            x = p * math.sin(lmd)
            y = p * math.cos(lmd)

            img_x = x - self.imgbasepos[0] # FIXME: wrong implementation,
            img_y = y - self.imgbasepos[1] #        not tested yet.

            (r,g,b) = img[img_x,img_y]
            intensity = (r/255.+g/255.+g/255.)
            return intensity

        data = \
            [
                [
                    [
                        [
                            getval(
                                self.basepos[0]+self.offset[0]+self.meshsize[0]*i,
                                self.basepos[1]+self.offset[1]+self.meshsize[1]*j,
                                k
                            )
                        ] for i in xrange(self.w)
                    ] for j in xrange(self.h)
                ] for k in xrange(self.d)
            ]

        return data