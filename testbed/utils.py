# -*- coding: utf-8 -*-
import sys

import collections
import numpy
import theano
import theano.tensor as T

from PySide import QtGui
sys.modules['PyQt4.QtGui'] = QtGui

from PIL import Image
from PIL.ImageQt import ImageQt

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
        return Image.fromarray(numpy.uint8(data*255))
    elif len(data.shape) == 3:
        return [ Image.fromarray(numpy.uint8(slice*255)) for slice in data ]
    else:
        raise TypeError("input data must be d-by-h-by-w ndarray")

def PILimageToQImage(pilimage):
    """converts a PIL image to QImage"""
    imageq = ImageQt(pilimage)
    qimage = QtGui.QImage(imageq)
    return qimage