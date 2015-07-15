import numpy
import pylab as plt

VIS_DEPTH = 0

class ObservationLocation:
    def __init__(self, vis, xy, fignum, onclose):
        def handle_close(event):
            onclose(event, self)

        self.vis = vis
        self.xy = xy
        self.fig = plt.figure(fignum)
        plt.clf()
        self.fig.canvas.mpl_connect('close_event', handle_close)
        self.ax = plt.subplot(111)

        x, y = self.xy
        data_y = [ data[VIS_DEPTH,y,x] for data in self.vis.data_y ]
        data_y_pred = [ data[VIS_DEPTH,y,x] for data in self.vis.data_y_pred ]
        self.plot_y = self.ax.plot(self.vis.data_x, data_y, 'b.-')
        self.plot_y_pred = self.ax.plot(self.vis.data_x, data_y_pred, 'r.-')

        plt.show()

    def update(self):
        x, y = self.xy
        data_y = [ data[VIS_DEPTH,y,x] for data in self.vis.data_y ]
        data_y_pred = [ data[VIS_DEPTH,y,x] for data in self.vis.data_y_pred ]
        self.plot_y[0].set_data(self.vis.data_x, data_y)
        self.plot_y_pred[0].set_data(self.vis.data_x, data_y_pred)
        self.ax.set_xlim(self.vis.data_x[0], self.vis.data_x[-1])
        self.ax.set_ylim(min(numpy.min(data_y), numpy.min(data_y_pred)), max(numpy.max(data_y), numpy.max(data_y_pred)))
        self.ax.autoscale_view(scalex=False,scaley=True)
        self.fig.canvas.draw()

class Visualizer:
    def __init__(self, w=10, h=10, xlim=30):
        # data
        self.data_x = []
        self.data_y = []
        self.data_y_pred = []

        # y
        self.fig_y = plt.figure(1)
        plt.clf()
        self.fig_y.canvas.mpl_connect('button_press_event', self.onclick)
        self.im_y = plt.imshow(numpy.zeros((w,h)), cmap=plt.cm.jet, vmin=0, vmax=1)
        self.colorbar_y = plt.colorbar()
        plt.show()

        # y_pred
        self.fig_y_pred = plt.figure(2)
        plt.clf()
        self.fig_y_pred.canvas.mpl_connect('button_press_event', self.onclick)
        self.im_y_pred = plt.imshow(numpy.zeros((w,h)), cmap=plt.cm.jet, vmin=0, vmax=1)
        self.colorbar_y_pred = plt.colorbar()
        plt.show()

        # observations
        self.observation_locations = []
        self.next_fignum = 3

        self.last_x = -1
        self.xlim=xlim

        self.ymin=0
        self.ymax=0

    def append(self, y, y_pred):
        assert isinstance(y, numpy.ndarray)
        assert isinstance(y_pred, numpy.ndarray)

        x = self.last_x + 1

        self.data_x.append(x)
        while self.xlim < len(self.data_x):
            self.data_x.pop(0)

        self.data_y.append(y)
        while self.xlim < len(self.data_y):
            self.data_y.pop(0)

        self.data_y_pred.append(y_pred)
        while self.xlim < len(self.data_y_pred):
            self.data_y_pred.pop(0)

        self.last_x = x
        self.update()

    def update(self):
        # y
        self.im_y.set_data(self.data_y[-1][VIS_DEPTH])
        self.fig_y.canvas.draw()

        # y_pred
        self.im_y_pred.set_data(self.data_y_pred[-1][VIS_DEPTH])
        self.fig_y_pred.canvas.draw()

        # timeseries
        for ol in self.observation_locations:
            ol.update()

    def addObservationLocation(self, xy):
        def handle_close(event, ol):
            self.observation_locations.remove(ol)

        ol = ObservationLocation(self, xy, self.next_fignum, handle_close)
        self.observation_locations.append(ol)
        self.next_fignum += 1

    def onclick(self, event):
        xy = (int(event.xdata), int(event.ydata))
        self.addObservationLocation(xy)
