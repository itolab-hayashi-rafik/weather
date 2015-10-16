# -*- coding: utf-8 -*-
import os
import glob
import time
import calendar
from datetime import datetime

import math
import numpy
import theano
import theano.tensor as T

import csv
from PIL import Image
import lrit

def tsxrange(begin='201408010000', end='201408312330', step='5'):
    '''
    timestamp range
    :param begin: フォーマット 'YYYYMMDDhhmm' の開始タイムスタンプ
    :param end: フォーマット 'YYYYMMDDhhmm' の終了タイムスタンプ
    :param step: フォーマット 'mm' のステップタイムスタンプ
    :return:
    '''
    tbegin = int(calendar.timegm(datetime.strptime(begin, '%Y%m%d%H%M').timetuple()))
    tend = int(calendar.timegm(datetime.strptime(end, "%Y%m%d%H%M").timetuple()))
    tstep = int(step)*60

    for t in xrange(tbegin, tend, tstep):
        stim = time.gmtime(t)
        tim = time.strftime("%Y%m%d%H%M", stim)
        yield tim

def tsrange(begin='201408010000', end='201408312330', step='5'):
    return [x for x in tsxrange(begin, end, step)]

def parse_radar(filepath, w, h, offset):
    with open(filepath) as f:
        reader = csv.reader(f)
        datetime    = next(reader)  # ヘッダーの読み飛ばし
        grid        = next(reader)  # ヘッダーの読み飛ばし
        header      = next(reader)  # ヘッダーの読み飛ばし
        location    = next(reader)  # ヘッダーの読み飛ばし
        range       = next(reader)  # ヘッダーの読み飛ばし

        n_rows, n_cols = map(lambda x: int(x), grid)

        w = w if 0 < w else n_cols
        h = h if 0 < h else n_rows
        w = w - offset[0] if n_cols < offset[0] + w else w
        h = h - offset[1] if n_rows < offset[1] + h else h

        data = numpy.zeros((1,h,w), dtype=theano.config.floatX)

        for timeline in reader:
            for row in xrange(n_rows):
                line = next(reader)
                if offset[1] <= row and row < h:
                    data[0,row,:] = map(lambda x: float(x), line[offset[0]:offset[0]+w])

    return data

def parse_satellite(filepath, w, h, d, offset, meshsize, basepos, lrit_settings):
    img = Image.open(filepath)

    def sec2degree(sec):
        return (sec/3600.)

    def getval(lon, lat, d):
        '''
        get the image intensity at (lon,lat) in secs
        '''
        x,y = lrit.lonlat2xy(
            prj_dir=lrit_settings['prj_dir'],
            prj_lon=lrit_settings['prj_lon'],
            lon=sec2degree(lon),
            lat=sec2degree(lat),
        )
        c,l = lrit.xy2cl(
            cfac=lrit_settings['CFAC'],
            lfac=lrit_settings['LFAC'],
            coff=lrit_settings['COFF'],
            loff=lrit_settings['LOFF'],
            x=x,
            y=y
        )

        (r,g,b) = img.getpixel((c,l))
        intensity = (r/255.+g/255.+g/255.)/3.
        return intensity

    d = -1 if lrit_settings['prj_dir'] == 'N' else 1

    data = numpy.zeros((d, h, w), dtype=theano.config.floatX)
    for k in xrange(d):
        for j in xrange(h):
            for i in xrange(w):
                data[k,j,i] = getval(
                    basepos[0] +   (offset[0]+i)*meshsize[0],
                    basepos[1] + d*(offset[1]+j)*meshsize[1],
                    k
                )

    return data

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
    def __init__(self, dir, w=0, h=0, offset=(0,0,0), begin='201408010000', end='201408312330', step='5'):
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

        self.i = -1
        self.i += offset[2]

        self.timestamps = tsrange(begin, end, step)

    def next(self):
        super(RadarGenerator, self).next()

        self.i = self.i + 1
        if self.i >= len(self.timestamps):
            print('RadarGenerator: no more files to read, last timestamp={0}'.format(self.timestamps[-1]))
            raise StopIteration

        timestamp = self.timestamps[self.i]
        filename = timestamp+'.csv'
        filepath = os.path.join(self.dir, filename)

        if not os.path.isfile(filepath):
            raise IOError("file not found: {0}".format(filepath))

        data = parse_radar(filepath, w=self.w, h=self.h, offset=self.offset)

        return data

class SatelliteGenerator(Generator):
    def __init__(self, dir, w=10, h=10, offset=(0,0,0), meshsize=(45,30), basepos=(491400,127800), begin='201408010000', end='201408312330', step='30'):
        '''

        :param dir:
        :param w:
        :param h:
        :param offset: offsets of (x, y, timestep)
        :param meshsize: the size of each cell in the grid (unit: sec)
        :param basepos: the lat long position of the northwest to extract (unit: sec)
        :return:
        '''
        # setting for POLAR(N,135) satellite images
        self.lrit_settings = {
            'prj_dir': 'N',
            'prj_lon': 135.,
            'CFAC': 99560944,
            'LFAC': 99440107,
            'COFF': 540,
            'LOFF': -420
        }

        super(SatelliteGenerator, self).__init__(w, h, 1)
        dir = os.path.join(os.path.dirname(__file__), dir)
        self.dir = dir
        self.offset = offset
        self.meshsize = meshsize
        self.basepos = basepos

        self.i = -1
        self.i += offset[2]

        self.timestamps = tsrange(begin, end, step)

    def next(self):
        super(SatelliteGenerator, self).next()

        self.i = self.i + 1
        if self.i >= len(self.timestamps):
            print('SatelliteGenerator: no more files to read, last timestamp={0}'.format(self.timestamps[-1]))
            raise StopIteration

        timestamp = self.timestamps[self.i]
        filename = timestamp+".jpg"
        filepath = os.path.join(self.dir, filename)

        if not os.path.isfile(filepath):
            raise IOError("file not found: {0}".format(filepath))

        data = parse_satellite(filepath, w=self.w, h=self.h, d=self.d,
                               offset=self.offset, meshsize=self.meshsize,
                               basepos=self.basepos, lrit_settings=self.lrit_settings)

        return data


def test_satellite_generator():
    gen = SatelliteGenerator('../eisei_PS01IR1', w=120, h=120)

    def dismissIOError(gen):
        while True:
            try:
                yield gen.next()
            except IOError as e:
                yield 'IOError {0}'.format(e)
            except StopIteration:
                break

    for i,sat in enumerate(dismissIOError(gen)):
        print('{0}: {1}'.format(i, sat))

if __name__ == '__main__':
    test_satellite_generator()