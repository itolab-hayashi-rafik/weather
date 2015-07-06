import time

import numpy
import theano
from theano import tensor as T

from dnn.SdA import SdA
from generator import SimpleGenerator
from visualizer import Visualizer

class TestBed:
    def __init__(self, window_size=1, m=1, r=2, batch_size=1, hidden_layers_sizes=[10]):
        self.r = r
        self.window_size = window_size
        self.n_batches = (self.window_size / batch_size)
        self.n_input = m*(r+1)
        self.n_output = m
        self.data = [[0 for j in xrange(m)] for i in xrange(window_size + (r+1))]
        self.x_value_pred = numpy.zeros((window_size, self.n_input), dtype=theano.config.floatX)
        self.x_value = numpy.zeros((window_size, self.n_input), dtype=theano.config.floatX)
        self.x = theano.shared(self.x_value, borrow=True)
        self.y_value = numpy.zeros((window_size, self.n_output), dtype=theano.config.floatX)
        self.y = theano.shared(self.y_value, borrow=True)

        numpy_rng = numpy.random.RandomState(89677)

        print '... building the model'
        # construct the stacked denoising autoencoder class
        self.sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=self.n_input,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=self.n_output
        )

        # retrieving functions
        self.pretraining_fns = self.sda.pretraining_functions(
            train_set_x=self.x,
            batch_size=batch_size
        )
        self.train_fn = self.sda.build_finetune_function(
            train_set_x=self.x,
            train_set_y=self.y,
            batch_size=batch_size,
        )
        self.predict_fn = self.sda.build_prediction_function()

    def supply(self, y):
        self.data.append(y)
        while self.window_size + (self.r+1) < len(self.data):
            self.data.pop(0)

        for i in xrange(self.x_value.shape[0]-1):
            self.x_value_pred[i] = self.x_value_pred[i+1]
            self.x_value[i] = self.x_value[i+1]
            self.y_value[i] = self.y_value[i+1]
        self.x_value_pred[-1] = [self.data[-1-i%(self.r+1)][int(i/(self.r+1))] for i in xrange(self.x_value.shape[1])]
        self.x_value[-1] = self.x_value_pred[-2]
        self.y_value[-1] = y

    def pretrain(self, pretraining_epochs, pretraining_lr):
        total_epochs = 0
        avg_cost = 0
        for i in xrange(self.sda.n_layers):
            corruption_level = i / 10.0
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                total_epochs = total_epochs + 1
                # go through the training set
                c = []
                for batch_index in xrange(self.n_batches):
                    c.append(self.pretraining_fns[i](index=batch_index, corruption=corruption_level, lr=pretraining_lr))
                minibatch_avg_cost = numpy.mean(c)
                avg_cost = avg_cost + minibatch_avg_cost
        avg_cost = avg_cost / total_epochs if 0 < total_epochs else 0
        return avg_cost

    def finetune(self, finetunning_epochs, finetunning_lr=0.1):
        done_looping = False
        epoch = 0
        avg_cost = 0
        while (epoch < finetunning_epochs) and (not done_looping):
            epoch = epoch + 1
            batch_idx = range(self.n_batches)
            numpy.random.shuffle(batch_idx)
            for minibatch_index in batch_idx:
                minibatch_avg_cost = self.train_fn(minibatch_index, lr=finetunning_lr)
                avg_cost = avg_cost + minibatch_avg_cost
        avg_cost = avg_cost / epoch if 0 < epoch else 0
        return avg_cost

    def predict(self):
        return self.predict_fn(self.x_value_pred)[-1]


def main(m=5, r=2, window_size=20, batch_size=2):
    gen = SimpleGenerator(num=m)
    bed = TestBed(m=m, r=r, window_size=window_size, batch_size=batch_size)
    vis = Visualizer()

    for i in xrange(10):
        bed.supply(gen.next())

    for i,y in enumerate(gen):
        if i % window_size == 0:
            # pretrain
            avg_cost = bed.pretrain(10, pretraining_lr=0.1)
            print("   pretrain cost: {}".format(avg_cost))

        # predict
        y_pred = bed.predict()
        print("{}: y={}, y_pred={}".format(i, y, y_pred))
        vis.append(y, y_pred)

        # finetune
        bed.supply(y)
        avg_cost = bed.finetune(10, finetunning_lr=0.1)
        # bed.finetune(100, finetunning_lr=0.01)
        # bed.finetune(100, finetunning_lr=0.001)
        print("   train cost: {}".format(avg_cost))
        time.sleep(.1)

if __name__ == '__main__':
    main(r=10)