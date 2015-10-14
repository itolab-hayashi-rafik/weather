__author__ = 'masayuki'
import os
import time
import calendar
from datetime import datetime

import numpy
from scipy.interpolate import RegularGridInterpolator

import gifmaker
from generator import SinGenerator, RadarGenerator, SatelliteGenerator

'''
weather dataset generator
'''

class GeneratorInterpolator(object):
    def __init__(self, generator, supply_num, method='linear'):
        '''
        :param generator: an instanceof Generator
        :param supply_num: how many frames to supply between frames
        '''
        self.generator = generator
        self.xs = numpy.asarray(range(0,generator.w))
        self.ys = numpy.asarray(range(0,generator.h))
        self.n_channels = generator.d
        self.supply_num = supply_num

        assert method in ['linear', 'nearest']
        self.method = method

        self.prev_instance = None
        self.next_instance = self.generator.next()
        self.supply_queue = []
        self.supply_f = [None for _ in xrange(self.n_channels)]

    def __iter__(self):
        return self

    def next(self):
        if len(self.supply_queue) == 0:
            # retrieve next available frame
            self.prev_instance = self.next_instance
            self.next_instance = None
            while self.next_instance is None:
                try:
                    self.next_instance = self.generator.next()
                except IOError:
                    self.next_instance = None
                ref = 0 if len(self.supply_queue) == 0 else self.supply_queue[-1]
                self.supply_queue += range(ref, ref+self.supply_num+1)

            # build interpolator
            ts = numpy.asarray([self.supply_queue[0], self.supply_queue[-1]])
            for channel in xrange(self.n_channels):
                seq = numpy.asarray([self.prev_instance[channel], self.next_instance[channel]])
                self.supply_f[channel] = RegularGridInterpolator((ts,self.ys,self.xs), seq, method=self.method)

            # remove redundant data
            self.supply_queue.pop(-1)

        t = self.supply_queue.pop(0)
        pts = numpy.array([[[(t,y,x) for x in self.xs] for y in self.ys]])
        frame = []
        for channel in xrange(self.n_channels):
            frame.append(self.supply_f[channel](pts)[0])

        return numpy.asarray(frame)



class WeatherDataGenerator(object):
    def __init__(self, seqnum=15000, seqdim=(10, 3, 16, 16), offset=(0,0,0), radar_dir='../radar', sat1_dir="../eisei_PS01IR1", sat2_dir="../eisei_PS01VIS",
                 begin='201408010000', end='201408312330', step='30'):
        self.generators = []
        # self.generators += [{
        #     'generator': RadarGenerator(radar_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
        #                                 begin=begin, end=end, step=step),
        #     'step': 1
        # }]
        self.generators += [{
            'generator': SatelliteGenerator(sat1_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
                                            begin=begin, end=end, step=step),
            'step': 1
        }]
        # self.generators += [{
        #     'generator': SatelliteGenerator(sat2_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
        #                                     begin=begin, end=end, step=step),
        #     'step': 1
        # }]

        self.seqnum = seqnum
        self.seqdim = seqdim

        self.setup()

    def setup(self):
        # initialize first frames with with zeros
        self.frames = [numpy.zeros((1,) + self.seqdim[2:], dtype=numpy.float32) for _ in self.generators]

        self.t = -1

    def __iter__(self):
        return self

    def next(self):
        self.t = self.t + 1
        err = False

        # generate frame by generators if necessary
        for i,entry in enumerate(self.generators):
            try:
                if self.t % entry['step'] == 0:
                    self.frames[i] = entry['generator'].next()
            except IOError as err:
                print('warning: IOError, {0}'.format(err))
                err = True

        # concatenate frames
        frame = numpy.concatenate(self.frames, axis=0)

        return frame if not err else None

def save_to_numpy_format(seq, input_seq_len, output_seq_len, path):
    # seq is of shape (n_samples, n_timesteps, n_feature_maps, height, width)
    assert 5 == seq.ndim
    assert input_seq_len + output_seq_len == seq.shape[1]
    dims = numpy.asarray([[seq.shape[2], seq.shape[3], seq.shape[4]]], dtype="int32")
    input_raw_data = seq.reshape((seq.shape[0] * seq.shape[1], seq.shape[2], seq.shape[3], seq.shape[4]))
    clips = numpy.zeros((2, seq.shape[0], 2), dtype="int32")
    clips[0, :, 0] = range(0, input_raw_data.shape[0], seq.shape[1])
    clips[0, :, 1] = input_seq_len
    clips[1, :, 0] = range(input_seq_len, input_raw_data.shape[0] + input_seq_len, seq.shape[1])
    clips[1, :, 1] = output_seq_len
    numpy.savez_compressed(path, dims=dims, input_raw_data=input_raw_data, clips=clips)

    print('output file is available at: {0}'.format(path))

def normalize(seqs):
    # seq is of shape (n_samples, n_timesteps, n_feature_maps, height, width)
    assert 5 == seqs.ndim
    zmax = numpy.max(seqs)
    zmin = numpy.min(seqs)
    normalized = (seqs - zmin) / (zmax - zmin)
    return zmin, zmax, normalized

def generator(seqnum=15000, seqdim=(10, 1, 120, 120), offset=(0,0,0), input_seq_len=5, output_seq_len=5, savedir='out'):
    '''
    generate sequences of weather data
    :param seqnum: How many sequences to generate
    :param seqdim: (n_timesteps, height, width)
    :param offset: (n_timesteps, top, left)
    :param steps: (step for radar, step for sat1, step for sat2)
    :param radar_dir:
    :param sat1_dir:
    :param sat2_dir:
    :param savedir:
    :return:
    '''

    gen = WeatherDataGenerator(seqnum=seqnum, seqdim=seqdim, offset=offset)
    frames = []

    def fill_frames():
        while len(frames) < seqdim[0]:
            frame = gen.next()
            if frame is None:
                del frames[:]
            else:
                frames.append(frame)

    # make output directory
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    print('... Generating sequences')
    fill_frames()
    seqs = numpy.zeros((seqnum,) + seqdim, dtype=numpy.float32)
    for i in range(seqnum):
        print('sequence {0} ...'.format(i)),
        seqs[i, :, :, :, :] = numpy.asarray(frames, dtype=numpy.float32)
        print('created')

        try:
            frames.pop(0)
            fill_frames()
        except StopIteration:
            seqs = seqs[:i]
            seqnum = i
            break

    print('done. {0} sequences in total'.format(seqnum))

    zmin, zmax, seqs = normalize(seqs)
    print('normlization:')
    print('  zmin={0}, zmax={1}'.format(zmin, zmax))

    if savedir is not '':
        for i in xrange(100):
            for d in xrange(seqdim[1]):
                outfile = savedir + "/" + str(i) + "-" + str(d) + ".gif"
                gifmaker.save_gif(seqs[i, :, d, :, :], outfile)
                print('  --> saved to {0}'.format(outfile))

    if savedir is not '':
        cut1 = int(seqnum*0.8)
        cut2 = int(seqnum*0.9)
        save_to_numpy_format(seqs[:cut1], input_seq_len, output_seq_len, savedir + "/dataset-train.npz")
        save_to_numpy_format(seqs[cut1:cut2], input_seq_len, output_seq_len, savedir + "/dataset-valid.npz")
        save_to_numpy_format(seqs[cut2:], input_seq_len, output_seq_len, savedir + "/dataset-test.npz")
    else:
        return zmin, zmax, seqs

def file_check(dir='../radar', begin="201408010000", end="201408312330", step=5):
    tbegin = int(calendar.timegm(datetime.strptime(begin, '%Y%m%d%H%M').timetuple()))
    tend = int(calendar.timegm(datetime.strptime(end, "%Y%m%d%H%M").timetuple()))
    tstep = int(step*60)

    for t in xrange(tbegin, tend, tstep):
        stim = time.gmtime(t)
        tim = time.strftime("%Y%m%d%H%M", stim)
        filename = tim + ".csv"
        filepath = dir + "/" + filename
        if not os.path.isfile(filepath):
            print('file '+filepath+' does not exist')

def test_intrp(supply_num=4):
    # g_radar = RadarGenerator("../radar", w=16, h=16, begin="201408010000", end="201408312330", step="5")
    # g_radar2= RadarGenerator("../radar", w=16, h=16, begin="201408010000", end="201408312330", step="5")
    # g_intrp = GeneratorInterpolator(g_radar2, supply_num, method='linear')
    g_sin1 = SinGenerator(w=5,h=5,d=1)
    g_sin2 = SinGenerator(w=5,h=5,d=1)
    g_intrp = GeneratorInterpolator(g_sin2, supply_num, method='linear')

    for i,data_intrp in enumerate(g_intrp):
        print(' Intrp:{0}'.format(data_intrp))
        if i % supply_num == 0:
            try:
                data_radar = g_sin1.next()
                print(' Radar:{0}'.format(data_radar))
                assert numpy.array_equal(data_intrp, data_radar)
            except IOError as e:
                print(' Radar:{0}'.format(e))
            print('\n')

if __name__ == '__main__':
    # file_check()
    # generator()
    test_intrp()