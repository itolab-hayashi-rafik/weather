from abc import ABCMeta, abstractmethod

class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def pretrain(self, dataset, epochs=100, learning_rate=0.1, batch_size=1):
        return

    @abstractmethod
    def train(self, dataset, epochs=100, learning_rate=0.1, batch_size=1):
        return

    @abstractmethod
    def predict(self, dataset):
        return