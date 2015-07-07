# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import time
import string

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'

from PySide import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import numpy

from testbed import TestBed
from generator import Generator
from visualizer import Visualizer
import utils

class Worker(QtCore.QThread):

    started = QtCore.Signal()
    updated = QtCore.Signal(numpy.ndarray, numpy.ndarray, QtGui.QImage, QtGui.QImage)
    stopped = QtCore.Signal()

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.bed = None
        self.gen = None

        self.delay = 1.0
        self.stop_flg = False
        self.mutex = QtCore.QMutex()

    def setup(self, window_size=20, n=2, w=10, h=10, d=1, hidden_layers_sizes=[10], pretrain_step=20):
        self.bed = TestBed(window_size=window_size, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes)
        self.gen = Generator(w=w, h=h, d=d)
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
        self.pretrain_epochs = params['pretrain_epochs']
        self.pretrain_lr = params['pretrain_lr']
        self.finetune_epochs = params['finetune_epochs']
        self.finetune_lr = params['finetune_lr']

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stop_flg = True

    def run(self):
        print("Worker: started")
        with QtCore.QMutexLocker(self.mutex):
            self.stop_flg = False
        self.started.emit()

        for i,y in enumerate(self.gen):
            # original
            y_images = utils.generateImage(y)
            y_qimage = utils.PILimageToQImage(y_images[0])

            # predict
            y_pred = self.bed.predict()
            y_pred_images = utils.generateImage(y_pred)
            y_pred_qimage = utils.PILimageToQImage(y_pred_images[0])
            print("{}: y={}, y_pred={}".format(i, y, y_pred))

            self.bed.supply(y)

            if i % self.pretrain_step == 0:
                # pretrain
                avg_cost = self.bed.pretrain(self.pretrain_epochs, learning_rate=self.pretrain_lr)
                print("   pretrain cost: {}".format(avg_cost))
                pass

            # finetune
            avg_cost = self.bed.finetune(self.finetune_epochs, learning_rate=self.finetune_lr)
            print("   train cost: {}".format(avg_cost))

            self.updated.emit(y, y_pred, y_qimage, y_pred_qimage)

            time.sleep(self.delay)

            if self.stop_flg:
                print(' --- iteration end ---')
                break

        self.stopped.emit()

class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.vis = Visualizer()
        self.canvas = FigureCanvas(self.vis.getFigure())

        self.scene = QtGui.QGraphicsScene(self)
        self.grview = QtGui.QGraphicsView(self.scene, self)
        self.grview.scale(10.0,10.0)
        self.scene_pred = QtGui.QGraphicsScene(self)
        self.grview_pred = QtGui.QGraphicsView(self.scene_pred, self)
        self.grview_pred.scale(10.0,10.0)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)

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
        self.hidden_layer_sizes_line_edit = QtGui.QLineEdit('10,10,10')
        self.hidden_layer_sizes_line_edit.textChanged.connect(self.dnnChanged)

        self.input_form = QtGui.QFormLayout()
        self.input_form.addRow('Window SIze:', self.window_size_line_edit)
        self.input_form.addRow('w:', self.w_line_edit)
        self.input_form.addRow('h:', self.h_line_edit)
        self.input_form.addRow('d:', self.d_line_edit)
        self.input_form.addRow('n:', self.n_line_edit)
        self.input_form.addRow('Hidden Layer Sizes:', self.hidden_layer_sizes_line_edit)

        self.pretrian_epochs_line_edit = QtGui.QLineEdit('10')
        self.pretrian_epochs_line_edit.textChanged.connect(self.updateWorker)
        self.pretrain_lr_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.pretrain_lr_slider.setRange(1, 10)
        self.pretrain_lr_slider.setValue(1)
        self.pretrain_lr_slider.valueChanged.connect(self.updateWorker)
        self.finetune_epochs_line_edit = QtGui.QLineEdit('10')
        self.finetune_epochs_line_edit.textChanged.connect(self.updateWorker)
        self.finetune_lr_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.finetune_lr_slider.setRange(1, 10)
        self.finetune_lr_slider.setValue(1)
        self.finetune_lr_slider.valueChanged.connect(self.updateWorker)

        self.learn_form = QtGui.QFormLayout()
        self.learn_form.addRow('finetune_epoch', self.finetune_epochs_line_edit)
        self.learn_form.addRow('finetune_lr', self.finetune_lr_slider)
        self.learn_form.addRow('pretrain_epoch', self.pretrian_epochs_line_edit)
        self.learn_form.addRow('pretrain_lr', self.pretrain_lr_slider)

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
        layout.addWidget(self.canvas, 0, 0, 1, 2)
        layout.addWidget(self.grview, 1, 0)
        layout.addWidget(self.grview_pred, 1, 1)
        layout.addLayout(self.input_form, 2, 0)
        layout.addLayout(self.learn_form, 2, 1)
        layout.addWidget(self.slider, 3, 0)
        layout.addWidget(self.start_stop_button, 3, 1)
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

        window_size = string.atoi(self.window_size_line_edit.text())
        w = string.atoi(self.w_line_edit.text())
        h = string.atoi(self.h_line_edit.text())
        d = string.atoi(self.d_line_edit.text())
        n = string.atoi(self.n_line_edit.text())
        hidden_layers_sizes = self.hidden_layer_sizes_line_edit.text().split(',')
        hidden_layers_sizes = [string.atoi(i) for i in hidden_layers_sizes]

        if self.need_setup:
            self.worker.setup(window_size=window_size, n=n, w=w, h=h, d=d, hidden_layers_sizes=hidden_layers_sizes, pretrain_step=1)
            self.need_setup = False
        self.updateWorker()
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

    def updateGraphics(self, y, y_pred, y_qimage, y_pred_qimage):
        # refresh canvas
        self.vis.append(y[0,0,0].tolist(), y_pred[0,0,0].tolist())
        self.canvas.draw()
        y_qpixmap = QtGui.QPixmap.fromImage(y_qimage)
        y_pred_qpixmap = QtGui.QPixmap.fromImage(y_pred_qimage)
        self.scene.addPixmap(y_qpixmap)
        self.scene_pred.addPixmap(y_pred_qpixmap)

    def workerStopped(self):
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.clicked.connect(self.start)

    def getLearningParams(self):
        return {
            'pretrain_epochs': string.atoi(self.pretrian_epochs_line_edit.text()),
            'pretrain_lr': 1.0/pow(10, self.pretrain_lr_slider.value()),
            'finetune_epochs': string.atoi(self.finetune_epochs_line_edit.text()),
            'finetune_lr': 1.0/pow(10, self.finetune_lr_slider.value())
        }

    def getDelayValue(self):
        return self.slider.value() / 100.0

if __name__ == '__main__':
    worker = Worker()
    worker.setup()
    worker.setLearningParams({
        'pretrain_epochs': 10,
        'pretrain_lr': 0.1,
        'finetune_epochs': 10,
        'finetune_lr': 0.1
    })
    worker.run()

    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())