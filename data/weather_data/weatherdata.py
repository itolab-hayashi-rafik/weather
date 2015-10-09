__author__ = 'masayuki'
import os
import time
import calendar
from datetime import datetime

import numpy

import gifmaker
from generator import RadarGenerator, SatelliteGenerator

'''
weather dataset generator
'''

class WeatherDataGenerator(object):
    def __init__(self, seqnum=15000, seqdim=(10, 3, 16, 16), offset=(0,0,0), radar_dir='../radar', sat1_dir="../eisei_PS01IR1", sat2_dir="../eisei_PS01VIS",
                 begin='201408010000', end='201408312330', step='5'):
        self.generators = []
        self.generators += [{
            'generator': RadarGenerator(radar_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
                                        begin=begin, end=end, step=step),
            'step': 1
        }]
        # self.generators += [{
        #     'generator': SatelliteGenerator(sat1_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
        #                                     begin=begin, end=end, step=step),
        #     'step': 1
        # }]
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

def generator(seqnum=8741, seqdim=(20, 1, 120, 120), offset=(0,0,0), savedir='out'):
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

        frames.pop(0)
        fill_frames()

        if savedir is not '':
            if i < 100:
                for d in xrange(seqdim[1]):
                    outfile = savedir + "/" + str(i) + "-" + str(d) + ".gif"
                    gifmaker.save_gif(seqs[i, :, d, :, :], outfile)
                    print('  --> saved to {0}'.format(outfile))

    if savedir is not '':
        cut1 = int(seqnum*0.8)
        cut2 = int(seqnum*0.9)
        save_to_numpy_format(seqs[:cut1], 10, 10, savedir + "/dataset-train.npz")
        save_to_numpy_format(seqs[cut1:cut2], 10, 10, savedir + "/dataset-valid.npz")
        save_to_numpy_format(seqs[cut2:], 10, 10, savedir + "/dataset-test.npz")
    else:
        return seqs

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

if __name__ == '__main__':
    # file_check()
    generator()