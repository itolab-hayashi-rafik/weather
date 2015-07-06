import math
import random
import time
import numpy as np
import visualizer as plt


class Generator(object):
    def __init__(self):
        self.last_state = 0

    def __iter__(self):
        return self

    def next(self):
        raise StopIteration

class SimpleGenerator(Generator):
    def __init__(self, num=1):
        super(SimpleGenerator, self).__init__()
        self.num = num
        self.i = 0
        self.k = 0 # shape
        self.n = 0 # noise

    def setK(self, k):
        self.k = max(0, k)

    def setN(self, n):
        self.n = min(max(0, n), 1)

    def noise(self, x):
        return (random.random() * self.n)

    def itrgenerate(self, x, l):
        if l % 2 == 0:
            y = (1 + math.sin((x+l/2) / (self.k+1) / math.pi)) * 0.5
        else:
            y = (1 + math.cos((x+l/2) / (self.k+1) / math.pi)) * 0.5
        y += self.noise(x)
        return y

    def generate(self, x):
        arr = [0 for _ in xrange(self.num)]
        for l in xrange(self.num):
            arr[l] = self.itrgenerate(x, l)
        return arr

    def next(self):
        x = self.i
        self.i += 1
        return self.generate(x)


if __name__ == '__main__':
    try:
        vis = plt.Visualizer()
        gen = SimpleGenerator()
        for y in gen:
            print("{}".format(y))
            vis.append(y)
            time.sleep(0.5)
    except Exception as e:
        print(str(e))