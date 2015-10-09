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

def generate_sequence(seqdim, steps, g_radar, g_sat1, g_sat2):
    seq = numpy.zeros(seqdim, dtype=numpy.float32)
    radar = numpy.zeros((1,) + seqdim[2:], dtype=numpy.float32)
    sat1 = numpy.zeros((1,) + seqdim[2:], dtype=numpy.float32)
    sat2 = numpy.zeros((1,) + seqdim[2:], dtype=numpy.float32)
    for i in xrange(seqdim[0]):
        radar = g_radar.next() if i % steps[0] == 0 else radar
        sat1 = g_sat1.next() if i % steps[1] == 0 else sat1
        sat2 = g_sat2.next() if i % steps[2] == 0 else sat2
        seq[i, :, :, :] = numpy.concatenate([radar, sat1, sat2], axis=0)
    return numpy.asarray(seq, dtype=numpy.float64)

def generator(seqnum=15000, seqdim=(10, 3, 16, 16), offset=(0,0,0), steps=(1,3,3), radar_dir='../radar', sat1_dir="../eisei_PS01IR1", sat2_dir="../eisei_PS01VIS", savedir='out'):
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
    g_radar = RadarGenerator(radar_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]))
    g_sat1 = SatelliteGenerator(sat1_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]))
    g_sat2 = SatelliteGenerator(sat2_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]))

    # make output directory
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    print('... Generating sequences')
    seq = numpy.zeros((seqnum,) + seqdim, dtype=numpy.float32)
    for i in range(seqnum):
        print('sequence {0} ...'.format(i)),
        seq[i, :, :, :, :] = generate_sequence(seqdim, steps, g_radar, g_sat1, g_sat2)
        print('created')
        if savedir is not '':
            if i < 100:
                for d in xrange(seqdim[1]):
                    outfile = savedir + "/" + str(i) + "-" + str(d) + ".gif"
                    gifmaker.save_gif(seq[i, :, d, :, :], outfile)
                    print('  --> saved to {0}'.format(outfile))

    if savedir is not '':
        save_to_numpy_format(seq[:10000], 10, 10, savedir + "/moving-mnist-train.npz")
        save_to_numpy_format(seq[10000:12000], 10, 10, savedir + "/moving-mnist-valid.npz")
        save_to_numpy_format(seq[12000:15000], 10, 10, savedir + "/moving-mnist-test.npz")
    else:
        return seq

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
    file_check()
    # generator()