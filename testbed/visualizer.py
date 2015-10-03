import time
import numpy
import pylab as plt

from generator import SinGenerator, RadarGenerator

VIS_DEPTH = 0
LINE_DEFS = ['b.-', 'r.-', 'g.-']

class ImageMap(object):
    def __init__(self, w, h, fignum, onclick=None):
        super(ImageMap, self).__init__()

        self.w = w
        self.h = h

        self.fig = plt.figure(fignum)
        plt.clf()
        if onclick is not None:
            self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.im = plt.imshow(numpy.zeros((w,h)), cmap=plt.cm.jet, vmin=0, vmax=1)
        self.colorbar = plt.colorbar()
        self.fig.show()

    def update(self):
        self.im.set_data(self.mapdata)
        self.fig.canvas.draw()

    @property
    def mapdata(self):
        return numpy.zeros((self.w, self.h))

class LineGraph(object):
    def __init__(self, fignum, onclose=None):
        super(LineGraph, self).__init__()

        self.fig = plt.figure(fignum)
        plt.clf()
        if onclose is not None:
            self.fig.canvas.mpl_connect('close_event', onclose)
        self.ax = plt.subplot(111)

        self.plots = []
        for i, (x,y) in enumerate(self.xydata):
            self.plots.append(self.ax.plot(x, y, LINE_DEFS[i % len(LINE_DEFS)])) # FIXME: line color

        self.fig.show()

    def update(self):
        xmin, xmax = (None, None)
        ymin, ymax = (None, None)
        for i, (x,y) in enumerate(self.xydata):
            self.plots[i][0].set_data(x, y)
            xmin = min(n for n in [xmin, numpy.min(x)] if n is not None)
            xmax = max(n for n in [xmax, numpy.max(x)] if n is not None)
            ymin = min(n for n in [ymin, numpy.min(y)] if n is not None)
            ymax = max(n for n in [ymax, numpy.max(y)] if n is not None)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.autoscale_view(scalex=False,scaley=True)
        self.fig.canvas.draw()

    @property
    def xydata(self):
        return [([0], [0])]

class YMap(ImageMap):
    def __init__(self, get_data, w, h, fignum, onclick=None):
        self.get_data = get_data

        super(YMap, self).__init__(w, h, fignum, onclick)

    @property
    def mapdata(self):
        return self.get_data()

class ObservationLocationGraph(LineGraph):
    def __init__(self, x, y, x_preds, y_preds, xy, t_out, fignum, onclose=None):
        self.x = x
        self.y = y
        self.x_preds = x_preds
        self.y_preds = y_preds
        self.xy = xy
        self.t_out = t_out

        def handle_close(event):
            onclose(event, self)

        super(ObservationLocationGraph, self).__init__(fignum, handle_close)

    @property
    def xydata(self):
        px, py = self.xy
        x = self.x
        y = [ data[VIS_DEPTH, px, py] for data in self.y ]

        preds = [
            (
                self.x_preds[i],
                [y_pred[i][VIS_DEPTH, px, py] for y_pred in self.y_preds]
            ) for i in xrange(self.t_out)
        ]

        return [(x, y)] + preds

class LearningCurve(LineGraph):
    def __init__(self, costs_x, costs_y, fignum):
        self.costs_x = costs_x
        self.costs_y = costs_y

        super(LearningCurve, self).__init__(fignum)

    @property
    def xydata(self):
        data_x = self.costs_x
        data_train_cost = [ costs[0] for costs in self.costs_y ]
        data_valid_cost = [ costs[1] for costs in self.costs_y ]
        data_test_cost = [ costs[2] for costs in self.costs_y ]

        return [
            (data_x, data_train_cost),
            (data_x, data_valid_cost),
            (data_x, data_test_cost),
        ]

class Dataset(object):
    def __init__(self, t_out=1, xlim=30, clim=30):
        super(Dataset, self).__init__()

        self.t_out = t_out
        self.xlim = xlim
        self.clim = clim

        self.last_x = -1
        self.last_cost_x = -1

        # x
        self.x = []
        self.x_preds = [[] for i in xrange(t_out)]

        # y
        self.y = []
        self.y_preds = []

        # cost
        self.costs_x = []
        self.costs_y = []

    def _fixed_append(self, list, item, maxlen):
        list.append(item)
        while maxlen < len(list):
            list.pop(0)

    def append_data(self, y, y_preds):
        x = self.last_x + 1

        self._fixed_append(self.x, x, self.xlim)
        for i in xrange(self.t_out):
            self._fixed_append(self.x_preds[i], x+i, self.xlim)
        self._fixed_append(self.y, y, self.xlim)
        self._fixed_append(self.y_preds, y_preds, self.xlim)

        self.last_x = x

    def append_cost(self, train_cost, valid_cost, test_cost):
        cost_x = self.last_cost_x + 1

        costs = (train_cost, valid_cost, test_cost)

        self._fixed_append(self.costs_x, cost_x, self.clim)
        self._fixed_append(self.costs_y, costs, self.clim)

        self.last_cost_x = cost_x


class Visualizer:
    def __init__(self, w=10, h=10, t_out=1, xlim=30, clim=100):
        self.t_out = t_out

        fignum = 1

        # dataset
        self.ds = Dataset(t_out, xlim, clim)

        # y
        self.im_y = YMap(lambda: self.ds.y[-1][VIS_DEPTH], w, h, fignum, self.onclick)
        fignum += 1

        # y_preds
        self.im_y_preds = []
        for i in xrange(t_out):
            self.im_y_preds.append(YMap(lambda i=i: self.ds.y_preds[-1][i][VIS_DEPTH], w, h, fignum, self.onclick))
            fignum += 1

        # learning curve
        self.graph_lc = LearningCurve(self.ds.costs_x, self.ds.costs_y, fignum)
        fignum += 1

        # observations
        self.graph_ols = []
        self.next_fignum = fignum

    def append_data(self, y, y_preds):
        self.ds.append_data(y, y_preds)
        self.update()

    def append_cost(self, train_cost, valid_cost=None, test_cost=None):
        self.ds.append_cost(train_cost, valid_cost, test_cost)
        self.graph_lc.update()

    def update(self):
        # y
        self.im_y.update()

        # y_preds
        for im in self.im_y_preds:
            im.update()

        # observation locations
        for ol in self.graph_ols:
            ol.update()

    def addObservationLocation(self, xy):
        def handle_close(event, olg):
            if olg in self.graph_ols:
                self.graph_ols.remove(olg)

        ol = ObservationLocationGraph(self.ds.x, self.ds.y, self.ds.x_preds, self.ds.y_preds, xy, self.t_out, self.next_fignum, handle_close)
        self.graph_ols.append(ol)
        self.next_fignum += 1

    def onclick(self, event):
        xy = (int(event.xdata), int(event.ydata))
        self.addObservationLocation(xy)


if __name__ == '__main__':
    w = 28
    h = 28
    delay = 0.1
    gen = SinGenerator(w=w, h=h, d=1)
    # gen = RadarGenerator('../data/radar', w=w, h=h, left=0, top=80)
    vis = Visualizer(w=w, h=h, t_out=2)

    time.sleep(10)
    ytm1 = gen.next()
    for i,yt in enumerate(gen):
        print("{0}: max={1}".format(i,numpy.max(yt)))
        vis.append_data(yt, numpy.asarray([ytm1, yt]))
        vis.append_cost(1.0/float(i+1), 2.0/float(i+1), None)

        if i == 10:
            vis.addObservationLocation((0,0))

        time.sleep(delay)
        ytm1 = yt