import os
import math
import numpy
import theano
import theano.tensor as T
import pylab as plt
from PIL import Image

from testbed.generator import SinGenerator

def filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def compile_conv(d=3, W_value=None):
    input = T.tensor4(name='input')

    filter_shape=(d,d,9,9)
    fan_in = numpy.prod(filter_shape[1:])
    fan_out = numpy.prod(filter_shape[0] * numpy.prod(filter_shape[2:]))
    normalize = 3.0 / fan_in * fan_out

    # init weight
    if W_value is None:
        W_value = numpy.asarray(
            [
                [
                    [
                        [
                            ((1 + math.sin((5 + i - j) / math.pi / 1.0)) * 0.5) / normalize
                            for i in xrange(filter_shape[3])
                            ] for j in xrange(filter_shape[2])
                        ] for k in xrange(filter_shape[1])
                    ] for l in xrange(filter_shape[0])
                ],
            dtype=theano.config.floatX
        )
    W = theano.shared(W_value, name='W', borrow=True)

    # convolve
    out = T.nnet.conv2d(input, W, border_mode='full')
    bh, bw = W.shape[2]//2, W.shape[3]//2
    sh, sw = (0 if W.shape[2] % 2 == 1 else 0, 0 if W.shape[3] % 2 == 1 else 0)
    out = out[:, :, bh:sh-bh, bw:sw-bw]

    # compile
    f_out = theano.function([input], out, name='f_out')

    return f_out, W_value, normalize

def generate_conv(image_file='data/Lenna.png', out_dir='out', W_value=None):
    # load image
    img_in = Image.open(image_file)
    img_in = img_in.resize((img_in.size[0]/1,img_in.size[1]/1))
    input = numpy.asarray(img_in, dtype=theano.config.floatX) / 256.
    w, h, d = input.shape
    input = theano.shared(input, borrow=True).dimshuffle('x',2,1,0)

    # compile conv
    f, W_value, normalize = compile_conv(d=d, W_value=W_value)

    # calculate the output
    out = f(input)
    out = theano.shared(out).dimshuffle(0,3,2,1).eval()

    img_in.save('{0}/{1}_in.png'.format(out_dir, filename(image_file)))
    # plt.imshow(img_in)

    for i in xrange(W_value.shape[0]):
        W_out_value = theano.shared(W_value[i]).dimshuffle(2,1,0).eval()
        img_filter = Image.fromarray(numpy.uint8(W_out_value * normalize * 255))
        img_filter.save('{0}/{1}_filter{2}.png'.format(out_dir, filename(image_file), i))
        # plt.imshow(img_filter)

    img_out = Image.fromarray(numpy.uint8(out[0] * 255))
    img_out.save('{0}/{1}_out.png'.format(out_dir, filename(image_file)))
    # plt.imshow(img_out)

    # plt.show()


if __name__ == '__main__':
    generate_conv()