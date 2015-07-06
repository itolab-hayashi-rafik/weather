import collections
import numpy
import theano

from PIL import Image

def ndarray(x):
    '''
    returns ndarray of x
    :param x:
    :return:
    '''
    if isinstance(x, numpy.ndarray):
        return x
    return numpy.asarray(x, dtype=theano.config.floatX)


def flatten(list):
    '''
    flatten a list
    :param list:
    :return:
    '''
    i = 0
    while i < len(list):
        while isinstance(list[i], collections.Iterable):
            if not list[i]:
                list.pop(i)
                i -= 1
                break
            else:
                list[i:i + 1] = list[i]
        i += 1
    return list

def generateImage(data):
    '''

    :param data:
    :return:
    '''
    assert isinstance(data, numpy.ndarray)

    if len(data.shape) == 2:
        return Image.fromarray(numpy.uint8(data))
    elif len(data.shape) == 3:
        return [ Image.fromarray(numpy.uint8(slice)) for slice in data ]
    else:
        raise TypeError("input data must be d-by-h-by-w ndarray")