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

def interpolate(generator, supply_num, method='linear'):
    '''
    Interpolate a generator
    :param generator: an instance of Generator
    :param supply_num: number of frames to supply between each generation
    :param method: 'linear' or 'nearest'
    :return:
    '''
    assert generator is not None
    assert supply_num >= 0

    xs = numpy.asarray(range(0,generator.w))
    ys = numpy.asarray(range(0,generator.h))
    n_channels = generator.d
    assert method in ['linear', 'nearest']

    prev_instance = None
    next_instance = generator.next()
    supply_queue = []
    supply_f = [None for _ in xrange(n_channels)]

    try:
        while True:
            if len(supply_queue) == 0:
                # retrieve next available frame
                prev_instance = next_instance
                next_instance = None
                while next_instance is None:
                    try:
                        next_instance = generator.next()
                    except IOError:
                        next_instance = None
                    ref = 0 if len(supply_queue) == 0 else supply_queue[-1]+1
                    supply_queue += range(ref, ref+supply_num+1)

                # build interpolator
                ts = numpy.asarray([supply_queue[0], supply_queue[-1]+1])
                for channel in xrange(n_channels):
                    seq = numpy.asarray([prev_instance[channel], next_instance[channel]])
                    supply_f[channel] = RegularGridInterpolator((ts,ys,xs), seq, method=method)

            t = supply_queue.pop(0)
            pts = numpy.array([[[(t,y,x) for x in xs] for y in ys]])
            frame = []
            for channel in xrange(n_channels):
                frame.append(supply_f[channel](pts)[0])

            yield numpy.asarray(frame)
    except StopIteration:
        yield prev_instance

class WeatherDataGenerator(object):
    def __init__(self, seqnum=15000, seqdim=(10, 3, 16, 16), offset=(0,0,0), radar_dir='../radar', sat1_dir="../eisei_PS01IR1", sat2_dir="../eisei_PS01VIS",
                 begin='201408010000', end='201408312330', step=5, method='linear'):
        self.generators = []
        self.generators += [{
            'generator': RadarGenerator(radar_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
                                        begin=begin, end=end, step='5'),
            'step': 5
        }]
        self.generators += [{
            'generator': SatelliteGenerator(sat1_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
                                            begin=begin, end=end, step='30'),
            'step': 30
        }]
        # self.generators += [{
        #     'generator': SatelliteGenerator(sat2_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
        #                                     begin=begin, end=end, step='30'),
        #     'step': 30
        # }]

        self.seqnum = seqnum
        self.seqdim = seqdim
        self.step = step
        self.method = method

        self.setup()

    def setup(self):
        # interpolate generators
        self._generators = []
        for entry in self.generators:
            generator = interpolate(entry['generator'], (entry['step']/self.step-1), method=self.method)
            self._generators.append(generator)

        # initialize first frames with with zeros
        self.frames = [numpy.zeros((1,) + self.seqdim[2:], dtype=numpy.float32) for _ in self.generators]

        self.t = -1

    def __iter__(self):
        return self

    def next(self):
        self.t = self.t + 1

        # generate frame by generators if necessary
        for i,generator in enumerate(self._generators):
            self.frames[i] = generator.next()

        # concatenate frames
        frame = numpy.concatenate(self.frames, axis=0)

        return frame

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
    zmax = numpy.max(seqs)
    zmin = numpy.min(seqs)
    normalized = (seqs - zmin) / (zmax - zmin)
    return zmin, zmax, normalized

def generator(seqnum=15000, seqdim=(10, 2, 120, 120), offset=(0,0,0), step=5, input_seq_len=5, output_seq_len=5, savedir='out'):
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

    gen = WeatherDataGenerator(seqnum=seqnum, seqdim=seqdim, offset=offset, step=step)
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

    zmins = [None for _ in xrange(seqdim[1])]
    zmaxs = [None for _ in xrange(seqdim[1])]
    for channel in xrange(seqdim[1]):
        zmin, zmax, seqs[:,:,channel,:,:] = normalize(seqs[:,:,channel,:,:])
        zmins[channel] = zmin
        zmaxs[channel] = zmax
        print('normlization (channel {0}):'.format(channel))
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
        return zmins, zmaxs, seqs

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

def test_intrp(supply_num=0):
    g_target = RadarGenerator("../radar", w=5, h=5, begin="201408010000", end="201408312330", step="5")
    g_target2= RadarGenerator("../radar", w=5, h=5, begin="201408010000", end="201408312330", step="5")
    g_intrp = interpolate(g_target2, supply_num, method='linear')
    # g_target = SinGenerator(w=5,h=5,d=1)
    # g_target2= SinGenerator(w=5,h=5,d=1)
    # g_intrp = interpolate(g_target2, supply_num, method='linear')

    for i,data_intrp in enumerate(g_intrp):
        print(' Intrp:{0}'.format(data_intrp))
        if numpy.mod(i,supply_num+1) == 0:
            try:
                data_radar = g_target.next()
                print(' Radar:{0}'.format(data_radar))
                assert numpy.array_equal(data_intrp, data_radar)
            except IOError as e:
                print(' Radar:{0}'.format(e))

def test_weather_data_generator():
    gen = WeatherDataGenerator(step=1)

    for i,data in enumerate(gen):
        print('{0}, {1}'.format(i,data))

if __name__ == '__main__':
    # file_check()
    generator()
    # test_intrp()
    # test_weather_data_generator()