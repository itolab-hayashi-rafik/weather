__author__ = 'masayuki'
import os
import time
import calendar
from datetime import datetime

import numpy
from scipy.interpolate import RegularGridInterpolator

import gifmaker
from generator import SinGenerator, RadarGenerator, SatelliteGenerator, Himawari8Generator

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

def skip(generator, skip_num):
    '''
    Skip a generator
    :param generator: an instance of Generator
    :param skip_num: number of frames to skip between each generation
    :return:
    '''
    gen = interpolate(generator, 0)
    try:
        while True:
            yield gen.next()
            for _ in xrange(skip_num):
                try:
                    gen.next()
                except Exception as e:
                    print("Warning: skipped empty data with the error"+str(e))
    except StopIteration:
        pass

class WeatherDataGenerator(object):
    def __init__(self, seqnum=15000, seqdim=(10, 3, 16, 16), offset=(0,0,0), radar_dir='../radar', sat1_dir="../eisei_PS01IR1", sat2_dir="../eisei_PS01VIS", himawari8_dir='../himawari8',
                 begin='201408010000', end='201408312330', step=5, method='linear', mode='grayscale'):
        self.generators = []
        self.generators += [{
            'generator': RadarGenerator(radar_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
                                        begin=begin, end=end, step='5'),
            'step': 5
        }]
        # self.generators += [{
        #     'generator': SatelliteGenerator(sat1_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
        #                                     begin=begin, end=end, step='30', mode=mode),
        #     'step': 30
        # }]
        # self.generators += [{
        #     'generator': SatelliteGenerator(sat2_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
        #                                     begin=begin, end=end, step='30', mode=mode),
        #     'step': 30
        # }]
        self.generators += [{
            'generator': Himawari8Generator(himawari8_dir, w=seqdim[-1], h=seqdim[-2], offset=(offset[2], offset[1], offset[0]),
                                            begin=begin, end=end, step='0500'),
            'step': 5
        }]

        self.seqnum = seqnum
        self.seqdim = seqdim
        self.step = step
        self.method = method

        self.setup()

    def setup(self):
        # interpolate generators
        self._generators = []
        for entry in self.generators:
            if entry['step'] > self.step:
                generator = interpolate(entry['generator'], int(entry['step']/self.step-1), method=self.method)
            elif entry['step'] == self.step:
                generator = entry['generator']
            else:
                generator = skip(entry['generator'], int(self.step/entry['step']-1))
            self._generators.append(generator)

        # initialize first frames with with zeros
        self.frames = [numpy.zeros(entry['generator'].dim, dtype=numpy.float32) for entry in self.generators]

        self.t = -1

    def __iter__(self):
        return self

    def next(self):
        self.t = self.t + 1
        hasError = False

        # generate frame by generators if necessary
        for i,generator in enumerate(self._generators):
            try:
                self.frames[i] = generator.next()
            except StopIteration:
                raise StopIteration
            except Exception as e:
                print('Error: generator['+str(i)+'] raised an error '+str(e))
                self.frames[i] = numpy.nan
                hasError = True

        if hasError:
            raise ValueError('Some of the generators raised errors')

        # concatenate frames
        frame = numpy.concatenate(self.frames, axis=0)

        return frame

def save_to_numpy_format(seq, input_seq_len, output_seq_len, zmaxs, zmins, path):
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
    numpy.savez_compressed(path, dims=dims, input_raw_data=input_raw_data, clips=clips, zmaxs=zmaxs, zmins=zmins)

    print('output file is available at: {0}'.format(path))

def normalize(seqs):
    # seq is of shape (n_samples, n_timesteps, n_feature_maps, height, width)
    assert seqs.ndim == 5

    zmaxs = [None for channel in xrange(seqs.shape[2])]
    zmins = [None for channel in xrange(seqs.shape[2])]
    for channel in xrange(seqs.shape[2]):
        zmax = numpy.max(seqs)
        zmin = numpy.min(seqs)
        seqs[:,:,channel,:,:] = (seqs[:,:,channel,:,:] - zmin) / (zmax - zmin)
        zmaxs[channel] = zmax
        zmins[channel] = zmin
        print('normlization (channel {0}):'.format(channel))
        print('  zmin={0}, zmax={1}'.format(zmin, zmax))
    return zmins, zmaxs

def generator(seqnum, seqdim, offset, begin, end, step, input_seq_len, output_seq_len, mode):
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
    print('generator(): '+str(locals()))

    gen = WeatherDataGenerator(seqnum=seqnum, seqdim=seqdim, offset=offset, begin=begin, end=end, step=step, mode=mode)
    frames = []

    def fill_frames():
        while len(frames) < seqdim[0]:
            try:
                frame = gen.next()
            except StopIteration:
                raise StopIteration
            except:
                frame = None

            if frame is None:
                del frames[:]
            else:
                frames.append(frame)

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
    return seqs[:seqnum]

def generate(seqnum=15000, seqdim=(20, 2, 120, 120), offset=(0,0,0), begin='201408010000', end='201408312330', step=30, input_seq_len=10, output_seq_len=10, mode='grayscale', savedir='out'):
    # make output directory
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    seqs = generator(**locals())
    seqnum = seqs.shape[0]
    zmins, zmaxs = normalize(seqs)

    if savedir is not '':
        for i in xrange(100):
            for d in xrange(seqdim[1]):
                outfile = savedir + "/" + str(i) + "-" + str(d) + ".gif"
                gifmaker.save_gif(seqs[i, :, d, :, :], outfile)
                print('  --> saved to {0}'.format(outfile))

    if savedir is not '':
        cut1 = int(seqnum*0.8)
        cut2 = int(seqnum*0.9)
        save_to_numpy_format(seqs[:cut1], input_seq_len, output_seq_len, zmaxs, zmins, savedir + "/dataset-train.npz")
        save_to_numpy_format(seqs[cut1:cut2], input_seq_len, output_seq_len, zmaxs, zmins, savedir + "/dataset-valid.npz")
        save_to_numpy_format(seqs[cut2:], input_seq_len, output_seq_len, zmaxs, zmins, savedir + "/dataset-test.npz")
    else:
        return zmins, zmaxs, seqs

def concat_generate(genargs=[{}], input_seq_len=10, output_seq_len=10, savedir='out'):
    # make output directory
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    seqs = [None for i in genargs]
    for i,args in enumerate(genargs):
        print('--------------------------')
        print('Generating for dataset {0}'.format(i))
        print('--------------------------')

        seqs[i] = generator(**args)

    seqs = numpy.concatenate(seqs, axis=0)
    seqnum = seqs.shape[0]
    zmins, zmaxs = normalize(seqs)

    if savedir is not '':
        for i in xrange(100):
            for d in xrange(seqs.shape[2]):
                outfile = savedir + "/" + str(i) + "-" + str(d) + ".gif"
                gifmaker.save_gif(seqs[i, :, d, :, :], outfile)
                print('  --> saved to {0}'.format(outfile))

    if savedir is not '':
        cut1 = int(seqnum*0.8)
        cut2 = int(seqnum*0.9)
        save_to_numpy_format(seqs[:cut1], input_seq_len, output_seq_len, zmaxs, zmins, savedir + "/dataset-train.npz")
        save_to_numpy_format(seqs[cut1:cut2], input_seq_len, output_seq_len, zmaxs, zmins, savedir + "/dataset-valid.npz")
        save_to_numpy_format(seqs[cut2:], input_seq_len, output_seq_len, zmaxs, zmins, savedir + "/dataset-test.npz")
    else:
        return zmins, zmaxs, seqs

def convert_to_multi_view(filepath):
    if not os.path.isfile(filepath):
        raise ValueError("file not found: "+filepath)

    filename, file_extension = os.path.splitext(filepath)

    f = numpy.load(filepath)
    input_raw_data = [f['input_raw_data'][:,[0],:,:], f['input_raw_data'][:,[1],:,:]]
    dims = [f['dims'][:], f['dims'][:]]
    dims[0][0][0] = 1
    dims[1][0][0] = 1
    clips = [f['clips'], f['clips']]
    numpy.savez('{0}-view0{1}'.format(filename, file_extension), clips=clips[0], dims=dims[0], input_raw_data=input_raw_data[0])
    numpy.savez('{0}-view1{1}'.format(filename, file_extension), clips=clips[1], dims=dims[1], input_raw_data=input_raw_data[1])

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
    # generator()
    # test_intrp()
    # test_weather_data_generator()

    seqnum =  15000
    w = 120
    h = 120

    # number of frames to input and output
    t_in = 10
    t_out = 10

    # dataset
    # begin='201408010000'
    # end='201408312330'

    # generate grayscale radar+sat1 of every 5 minutes with interpolator
    # generate(seqnum, (t_in+t_out, 2, h, w), begin=begin, end=end, step=5, input_seq_len=t_in, output_seq_len=t_out, mode='grayscale', savedir='out_radar_sat1')

    # generate grayscale radar+sat1 of every 30 minutes
    # generate(seqnum, (t_in+t_out, 2, h, w), begin=begin, end=end, step=30, input_seq_len=t_in, output_seq_len=t_out, mode='grayscale', savedir='out_radar_sat1_step30')

    # generate rgb radar+sat1 of every 5 minutes with interpolator
    # generate(seqnum, (t_in+t_out, 4, h, w), begin=begin, end=end, step=5, input_seq_len=t_in, output_seq_len=t_out, mode='rgb', savedir='out_radar_sat1_rgb')

    # generate grayscale radar+sat1 of every 30 minutes
    # generate(seqnum, (t_in+t_out, 4, h, w), begin=begin, end=end, step=30, input_seq_len=t_in, output_seq_len=t_out, mode='rgb', savedir='out_radar_sat1_rgb_step30')

    # generate grayscale radar+sat1 of every 30 minutes
    genargs = [
        {'seqnum': 15000,
         'seqdim': (t_in+t_out, 2, h, w),
         'offset': (0,0,0),
         'begin': '20151001000000',
         'end': '20151031235730',
         'step': 5,
         'input_seq_len': t_in,
         'output_seq_len': t_out,
         'mode': 'grayscale'},
    ]
    concat_generate(genargs, input_seq_len=t_in, output_seq_len=t_out, savedir='out_radar_himawari8_step5')

    # convert_to_multi_view('out_radar_himawari8_step5/dataset-train.npz')
    # convert_to_multi_view('out_radar_himawari8_step5/dataset-valid.npz')
    # convert_to_multi_view('out_radar_himawari8_step5/dataset-test.npz')