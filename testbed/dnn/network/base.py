from theano.configparser import config
from theano.tensor.basic import _multi
from theano.tensor.type import TensorType

ctensor5 = TensorType('complex64', ((False,) * 5))
ztensor5 = TensorType('complex128', ((False,) * 5))
ftensor5 = TensorType('float32', ((False,) * 5))
dtensor5 = TensorType('float64', ((False,) * 5))
btensor5 = TensorType('int8', ((False,) * 5))
wtensor5 = TensorType('int16', ((False,) * 5))
itensor5 = TensorType('int32', ((False,) * 5))
ltensor5 = TensorType('int64', ((False,) * 5))


def tensor5(name=None, dtype=None):
    """Return a symbolic 4-D variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False, False))
    return type(name)
tensor5s, ftensor5s, dtensor5s, itensor5s, ltensor5s = _multi(
    tensor5, ftensor5, dtensor5, itensor5, ltensor5)


class Network(object):
    def __init__(self):
        pass