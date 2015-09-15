import numpy
import theano
import theano.tensor as T

from base import Layer

class LinearRegression(Layer):
    def setup(self):
        super(LinearRegression, self).setup()
        self.y_pred = self.output

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            return T.mean(T.sqr(y-self.y_pred))
        else:
            raise NotImplementedError()
