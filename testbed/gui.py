# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import time

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'

from PySide import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import numpy

from testbed import TestBed
from generator import SinGenerator, RadarGenerator
from visualizer import Visualizer
import utils

class Worker(QtCore.QThread):

    started = QtCore.Signal()
    updated = QtCore.Signal(numpy.ndarray, numpy.ndarray)
    stopped = QtCore.Signal()

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.bed = None
        self.gen = None

        self.delay = 1.0
        self.stop_flg = False
        self.mutex = QtCore.QMutex()

    def setup(self, window_size=20, n=2, w=10, h=10, d=1, hidden_layers_sizes=[100], pretrain_step=20):
        self.bed = TestBed(window_size=window_size, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)
        self.gen = SinGenerator(w=w, h=h, d=1)
        # self.gen = RadarGenerator('../data/radar', w=w, h=h, left=0, top=80)
        self.vis = Visualizer(w=w, h=h)
        self.pretrain_step = pretrain_step

        # fill the window with data
        for i in xrange(window_size):
            y = self.gen.next()
            self.bed.supply(y)

    def setGeneratorParams(self, k, n):
        pass

    def setDelay(self, delay):
        self.delay = delay

    def setLearningParams(self, params):
        self.finetune_epochs = params['finetune_epochs']
        self.finetune_lr = params['finetune_lr']
        self.finetune_batch_size = params['finetune_batch_size']
        self.pretrain_epochs = params['pretrain_epochs']
        self.pretrain_lr = params['pretrain_lr']
        self.pretrain_batch_size = params['pretrain_batch_size']

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stop_flg = True

    def run(self):
        print("Worker: started")
        with QtCore.QMutexLocker(self.mutex):
            self.stop_flg = False
        self.started.emit()

        for i,y in enumerate(self.gen):
            # predict
            y_pred = self.bed.predict()
            #print("{}: y={}, y_pred={}".format(i, y, y_pred))

            self.bed.supply(y)
            self.vis.append(y, y_pred)

            if i % self.pretrain_step == 0 and 0 < self.pretrain_epochs:
                # pretrain
                avg_cost = self.bed.pretrain(self.pretrain_epochs, learning_rate=self.pretrain_lr, batch_size=self.pretrain_batch_size)
                print("   pretrain cost: {}".format(avg_cost))
                pass

            # finetune
            avg_cost = self.bed.finetune(self.finetune_epochs, learning_rate=self.finetune_lr, batch_size=self.finetune_batch_size)
            print("   train cost: {}".format(avg_cost))

            self.updated.emit(y, y_pred)

            time.sleep(self.delay)

            if self.stop_flg:
                print(' --- iteration end ---')
                break

        self.stopped.emit()

class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Form
        self.window_size_line_edit = QtGui.QLineEdit('10')
        self.window_size_line_edit.textChanged.connect(self.dnnChanged)
        self.w_line_edit = QtGui.QLineEdit('10')
        self.w_line_edit.textChanged.connect(self.dnnChanged)
        self.h_line_edit = QtGui.QLineEdit('10')
        self.h_line_edit.textChanged.connect(self.dnnChanged)
        self.d_line_edit = QtGui.QLineEdit('1')
        self.d_line_edit.textChanged.connect(self.dnnChanged)
        self.n_line_edit = QtGui.QLineEdit('2')
        self.n_line_edit.textChanged.connect(self.dnnChanged)
        self.hidden_layer_sizes_line_edit = QtGui.QLineEdit('3')
        self.hidden_layer_sizes_line_edit.textChanged.connect(self.dnnChanged)

        self.input_form = QtGui.QFormLayout()
        self.input_form.addRow('Window SIze:', self.window_size_line_edit)
        self.input_form.addRow('width:', self.w_line_edit)
        self.input_form.addRow('height:', self.h_line_edit)
        self.input_form.addRow('depth:', self.d_line_edit)
        self.input_form.addRow('n:', self.n_line_edit)
        self.input_form.addRow('Hidden Layer Sizes:', self.hidden_layer_sizes_line_edit)

        self.pretrain_epochs_line_edit = QtGui.QLineEdit('0')
        self.pretrain_epochs_line_edit.textChanged.connect(self.updateWorker)
        self.pretrain_lr_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.pretrain_lr_slider.setRange(1, 10)
        self.pretrain_lr_slider.setValue(1)
        self.pretrain_lr_slider.valueChanged.connect(self.updateWorker)
        self.pretrain_batch_size_line_edit = QtGui.QLineEdit('1')
        self.pretrain_batch_size_line_edit.textChanged.connect(self.updateWorker)
        self.finetune_epochs_line_edit = QtGui.QLineEdit('10')
        self.finetune_epochs_line_edit.textChanged.connect(self.updateWorker)
        self.finetune_lr_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.finetune_lr_slider.setRange(1, 10)
        self.finetune_lr_slider.setValue(1)
        self.finetune_lr_slider.valueChanged.connect(self.updateWorker)
        self.finetune_batch_size_line_edit = QtGui.QLineEdit('1')
        self.finetune_batch_size_line_edit.textChanged.connect(self.updateWorker)

        self.learn_form = QtGui.QFormLayout()
        self.learn_form.addRow('finetune_epoch', self.finetune_epochs_line_edit)
        self.learn_form.addRow('finetune_lr', self.finetune_lr_slider)
        self.learn_form.addRow('finetune_batch_size', self.finetune_batch_size_line_edit)
        self.learn_form.addRow('pretrain_epoch', self.pretrain_epochs_line_edit)
        self.learn_form.addRow('pretrain_lr', self.pretrain_lr_slider)
        self.learn_form.addRow('pretrain_batch_size', self.pretrain_batch_size_line_edit)

        # A slider to control the plot delay
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 99)
        self.slider.setValue(25)
        self.slider.valueChanged.connect(self.updateWorker)

        # A slider to control K
        self.k_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.k_slider.setRange(0,100)
        self.k_slider.setValue(0)
        self.k_slider.valueChanged.connect(self.updateWorker)
        self.n_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.n_slider.setRange(0,100)
        self.n_slider.setValue(0)
        self.n_slider.valueChanged.connect(self.updateWorker)

        # Just some button connected to `plot` method
        self.start_stop_button = QtGui.QPushButton('Start')
        self.start_stop_button.clicked.connect(self.start)

        # set the layout
        layout = QtGui.QGridLayout()
        # layout.addWidget(self.toolbar)
        layout.addLayout(self.input_form, 0, 0)
        layout.addLayout(self.learn_form, 0, 1)
        layout.addWidget(self.slider, 1, 0)
        layout.addWidget(self.start_stop_button, 1, 1)
        self.setLayout(layout)

        # setup worker
        self.need_setup = True
        self.worker = Worker()

        # setup event dispatchers
        self.worker.started.connect(self.workerStarted)
        self.worker.updated.connect(self.updateGraphics)
        self.worker.stopped.connect(self.workerStopped)

    def start(self):
        self.start_stop_button.setText('Stop')
        self.start_stop_button.setEnabled(False)

        window_size = int(self.window_size_line_edit.text())
        w = int(self.w_line_edit.text())
        h = int(self.h_line_edit.text())
        d = int(self.d_line_edit.text())
        n = int(self.n_line_edit.text())
        hidden_layers_sizes = self.hidden_layer_sizes_line_edit.text().split(',')
        hidden_layers_sizes = [int(i) for i in hidden_layers_sizes]

        # self.vis = Visualizer(w=w, h=h)

        if self.need_setup:
            self.worker.setup(window_size=window_size, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes, pretrain_step=1)
            self.need_setup = False
        self.updateWorker()
        # self.worker.run() # use this for debugging
        self.worker.start()

    def stop(self):
        self.start_stop_button.setText('Start')
        self.start_stop_button.setEnabled(False)
        self.worker.stop()

    def dnnChanged(self):
        self.need_setup = True

    def updateWorker(self):
        # self.worker.setGeneratorParams(self.getKValue(), self.getNValue())
        self.worker.setDelay(self.getDelayValue())
        self.worker.setLearningParams(self.getLearningParams())

    def workerStarted(self):
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.clicked.connect(self.stop)

    def updateGraphics(self, y, y_pred):
        # self.vis.append(y, y_pred)
        pass

    def workerStopped(self):
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.clicked.connect(self.start)

    def getLearningParams(self):
        return {
            'finetune_epochs': int(self.finetune_epochs_line_edit.text()),
            'finetune_lr': 1.0/pow(10, self.finetune_lr_slider.value()),
            'finetune_batch_size': int(self.finetune_batch_size_line_edit.text()),
            'pretrain_epochs': int(self.pretrain_epochs_line_edit.text()),
            'pretrain_lr': 1.0/pow(10, self.pretrain_lr_slider.value()),
            'pretrain_batch_size': int(self.pretrain_batch_size_line_edit.text()),
        }

    def getDelayValue(self):
        return self.slider.value() / 100.0

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())