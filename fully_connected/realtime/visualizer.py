import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, xlim=30):
        self.fig = plt.figure()
        self.axs = None
        self.plots = None
        self.xs = None
        self.ys = None
        self.last_x = -1
        self.xlim=xlim

        self.ymin=0
        self.ymax=0

        self.m = -1

    def getFigure(self):
        return self.fig

    def initDraw(self, m):
        self.xs = ( [], [] )	# xdata, pred_xdata
        self.ys = [ ([], []) for _ in xrange(m) ] # ydata, pred_ydata
        self.axs = []
        self.plots = []

        # create axes and plots
        for i in xrange(m):
            ax = self.fig.add_subplot(m,1,i+1)
            plot_y = ax.plot(self.xs[0], self.ys[i][0], "b.-")[0]
            plot_y_pred = ax.plot(self.xs[1], self.ys[i][1], "r.-")[0]
            self.axs.append(ax)
            self.plots.append([plot_y, plot_y_pred])

        self.m = m
        plt.draw()

    def append(self, y, y_pred=None):
        if len(y) != self.m:
            self.initDraw(len(y))

        x = self.last_x + 1
        xdata, x_preddata = self.xs
        xdata.append(x)
        if y_pred is not None:
            x_preddata.append(x)
        self.last_x = x

        for i in xrange(len(self.axs)):
            ydata, pred_ydata = self.ys[i]
            ydata.append(y[i])
            self.ymin = min(self.ymin, y[i])
            self.ymax = max(self.ymax, y[i])
            if y_pred is not None:
                pred_ydata.append(y_pred[i])
                self.ymin = min(self.ymin, y_pred[i])
                self.ymax = max(self.ymax, y_pred[i])

        self.update()

    def update(self):
        for i in xrange(len(self.axs)):
            self.plots[i][0].set_data(self.xs[0], self.ys[i][0])
            self.plots[i][1].set_data(self.xs[1], self.ys[i][1])
            self.axs[i].set_xlim(self.last_x-self.xlim, self.last_x)
            self.axs[i].set_ylim(self.ymin, self.ymax)
            self.axs[i].autoscale_view(scalex=False,scaley=True)

        plt.draw()