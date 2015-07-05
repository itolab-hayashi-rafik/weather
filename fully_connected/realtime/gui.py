import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import time

import numpy
import theano
from theano import tensor as T

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'

from PySide import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from generator import SimpleGenerator
from test_bed import TestBed
from visualizer import Visualizer

class Worker(QtCore.QThread):

    started = QtCore.Signal()
    updated = QtCore.Signal()
    stopped = QtCore.Signal()

    def __init__(self, vis, parent=None):
        super(Worker, self).__init__(parent)
        self.vis = vis
        self.stop_flg = False
        self.mutex = QtCore.QMutex()
        pass

    def setup(self, m=2, r=2, window_size=20, batch_size=1, pretrain_step=20, delay=0.1):
        self.bed = TestBed(m=m, r=r, window_size=window_size, batch_size=batch_size)
        self.gen = SimpleGenerator(num=m)
        self.pretrain_step = pretrain_step
        self.delay = delay
        self.stop_flg = False

    def setDelay(self, delay):
        self.delay = delay

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stop_flg = True

    def run(self):
        self.started.emit()

        for i,y in enumerate(self.gen):
            if i % self.pretrain_step == 0:
                # pretrain
                avg_cost = self.bed.pretrain(10, pretraining_lr=0.1)
                print("   pretrain cost: {}".format(avg_cost))

            # predict
            y_pred = self.bed.predict()
            print("{}: y={}, y_pred={}".format(i, y, y_pred))
            self.vis.append(y, y_pred)

            # finetune
            self.bed.supply(y)
            avg_cost = self.bed.finetune(10, finetunning_lr=0.1)
            # bed.finetune(100, finetunning_lr=0.01)
            # bed.finetune(100, finetunning_lr=0.001)
            print("   train cost: {}".format(avg_cost))
            time.sleep(self.delay)

            self.updated.emit()

            if self.stop_flg:
                break

        self.stopped.emit()

class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # visualizer
        self.vis = Visualizer(xlim=30)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.vis.getFigure())

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)

        # A slider to control the plot delay
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 99)
        self.slider.setValue(25)
        self.slider.valueChanged.connect(self.updateWorker)

        # Just some button connected to `plot` method
        self.start_stop_button = QtGui.QPushButton('Start')
        self.start_stop_button.clicked.connect(self.start)

        # set the layout
        layout = QtGui.QGridLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 0, 0, 1, 2)
        layout.addWidget(self.slider, 1, 0)
        layout.addWidget(self.start_stop_button, 1, 1)
        self.setLayout(layout)

        # setup worker
        self.worker = Worker(self.vis)

        # setup event dispatchers
        self.worker.started.connect(self.workerStarted)
        self.worker.updated.connect(self.updateFigure)
        self.worker.stopped.connect(self.workerStopped)

    def start(self):
        self.start_stop_button.setText('Stop')
        self.start_stop_button.setEnabled(False)
        self.worker.setup(delay=self.getDelayValue())
        self.worker.start()

    def stop(self):
        self.start_stop_button.setText('Start')
        self.start_stop_button.setEnabled(False)
        self.worker.stop()

    def updateWorker(self):
        self.worker.setDelay(self.getDelayValue())

    def workerStarted(self):
        self.start_stop_button.setEnabled(True)

    def updateFigure(self):
        # refresh canvas
        self.canvas.draw()

    def workerStopped(self):
        self.start_stop_button.setEnabled(True)

    def getDelayValue(self):
        return self.slider.value() / 100.0



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())