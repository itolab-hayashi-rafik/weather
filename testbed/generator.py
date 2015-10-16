# -*- coding: utf-8 -*-
import os
import glob

import math
import numpy
import theano
import theano.tensor as T

import csv
from PIL import Image
import lrit

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

        self.i = self.i + 1
        if self.i >= len(self.files):
            raise StopIteration

        data = None

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

            data = numpy.zeros((1,h,w), dtype=theano.config.floatX)

            for timeline in reader:
                for row in xrange(n_rows):
                    line = next(reader)
                    if self.offset[1] <= row and row < h:
                        data[0,row,:] = map(lambda x: float(x), line[self.offset[0]:self.offset[0]+w])

        return data / 100.0

class SatelliteGenerator(Generator):
    def __init__(self, dir, w=10, h=10, offset=(0,0,0), meshsize=(45,30), basepos=(491400,127800)):
        '''

        :param dir:
        :param w:
        :param h:
        :param offset: offsets of (x, y, timestep)
        :param meshsize: the size of each cell in the grid (unit: sec)
        :param basepos: the lat long position of the northwest (unit: sec)
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

        cwd = os.getcwd()
        os.chdir(dir)
        self.files = glob.glob('*.jpg')
        self.files.sort()
        os.chdir(cwd)

        self.i = -1
        self.i += offset[2]

    def next(self):
        super(SatelliteGenerator, self).next()

        self.i = self.i + 1
        if self.i >= len(self.files):
            raise StopIteration

        file = self.files[self.i]
        filepath = os.path.join(self.dir, file)
        img = Image.open(filepath)

        def sec2degree(sec):
            return (sec/3600.)

        def getval(lon, lat, d):
            '''
            get the image intensity at (lon,lat) in secs
            '''
            x,y = lrit.lonlat2xy(
                prj_dir=self.lrit_settings['prj_dir'],
                prj_lon=self.lrit_settings['prj_lon'],
                lon=sec2degree(lon),
                lat=sec2degree(lat),
            )
            c,l = lrit.xy2cl(
                cfac=self.lrit_settings['CFAC'],
                lfac=self.lrit_settings['LFAC'],
                coff=self.lrit_settings['COFF'],
                loff=self.lrit_settings['LOFF'],
                x=x,
                y=y
            )

            (r,g,b) = img.getpixel((c,l))
            intensity = (r/255.+g/255.+g/255.)/3.
            return intensity

        data = numpy.zeros((self.d, self.h, self.w), dtype=theano.config.floatX)
        for k in xrange(self.d):
            for j in xrange(self.h):
                for i in xrange(self.w):
                    data[k,j,i] = getval(
                        self.basepos[0]+(self.offset[0]+i)*self.meshsize[0],
                        self.basepos[1]+(self.offset[1]+j)*self.meshsize[1],
                        k
                    )

        return data


def gen_dataset(t_in=5, w=10, h=10, offset=(0,0,0), t_out=15):
    '''
    generate dataset using RadarGenerator, SatelliteGenerator
    :return:
    '''

    DATA_WIDTH = 120  # width of the original csv data (radar)
    DATA_HEIGHT = 120 # height of the original csv data (radar)

    input_width = DATA_WIDTH-offset[0]
    input_height= DATA_HEIGHT-offset[1]

    # calculate patchsize
    patchsize = (int(input_width / w), int(input_height / h))
    n_patches = numpy.prod(patchsize)
    step = t_in + t_out

    # initialize generators
    g_radar = RadarGenerator("../data/radar", w=input_width, h=input_height, offset=offset)

    # initialize dataset
    data_x = []
    data_y = []

    # a function to append cropped data to lists
    def append_patches(lists, data):
        assert len(lists) == n_patches

        k = 0
        for j in xrange(patchsize[1]):
            for i in xrange(patchsize[0]):
                bound_x = (i*w, (i+1)*w)
                bound_y = (i*h, (i+1)*h)
                patch = data[:, bound_y[0]:bound_y[1], bound_x[0]:bound_x[1]]
                lists[k].append(patch)
                k += 1

    print('Begin generating dataset\n')

    # generate data
    for i,radar in enumerate(g_radar):
        print('[{0}]'.format(i)),

        if i % step == 0:
            inputs = [[] for _ in xrange(n_patches)]
            outputs = [[] for _ in xrange(n_patches)]

        if len(inputs[0]) < t_in:
            append_patches(inputs, radar)
        elif len(outputs[0]) < t_out:
            append_patches(outputs, radar)

        if i % step == step-1:
            for input,output in zip(inputs,outputs):
                data_x.append(input)
                data_y.append(output)
            print(' --> appended to dataset, {0} data in total'.format(len(data_x)))

    print('\nend generating dataset')
    print('{0} data in total'.format(len(data_x)))

    return numpy.asarray(data_x, dtype=theano.config.floatX), numpy.asarray(data_y, dtype=theano.config.floatX)


if __name__ == '__main__':
    print('generating dataset\n')
    outfile = 'dataset.npz'
    dataset = gen_dataset()
    numpy.savez(outfile, dataset=dataset)
    print('\ndone, output file: {0}'.format(outfile))