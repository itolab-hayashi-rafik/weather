import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import time
import string

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

    def setup(self, m=2, r=2, window_size=20, batch_size=1, hidden_layer_sizes=[10], pretrain_step=20):
        self.bed = TestBed(m=m, r=r, window_size=window_size, batch_size=batch_size, hidden_layers_sizes=hidden_layer_sizes)
        self.gen = SimpleGenerator(num=m)
        self.pretrain_step = pretrain_step

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
        self.stop_flg = False
        self.started.emit()

        for i,y in enumerate(self.gen):
            if i % self.pretrain_step == 0:
                # pretrain
                avg_cost = self.bed.pretrain(self.pretrain_epochs, pretraining_lr=self.pretrain_lr)
                print("   pretrain cost: {}".format(avg_cost))

            # predict
            y_pred = self.bed.predict()
            print("{}: y={}, y_pred={}".format(i, y, y_pred))
            self.vis.append(y, y_pred)

            # finetune
            self.bed.supply(y)
            avg_cost = self.bed.finetune(self.finetune_epochs, finetunning_lr=self.finetune_lr)
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

        # Form
        self.window_size_line_edit = QtGui.QLineEdit('10')
        self.m_line_edit = QtGui.QLineEdit('1')
        self.r_line_edit = QtGui.QLineEdit('2')
        self.hidden_layer_sizes_line_edit = QtGui.QLineEdit('10,10,10')

        self.input_form = QtGui.QFormLayout()
        self.input_form.addRow('Window SIze:', self.window_size_line_edit)
        self.input_form.addRow('m:', self.m_line_edit)
        self.input_form.addRow('r:', self.r_line_edit)
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

        # Just some button connected to `plot` method
        self.start_stop_button = QtGui.QPushButton('Start')
        self.start_stop_button.clicked.connect(self.start)

        # set the layout
        layout = QtGui.QGridLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 0, 0, 1, 2)
        layout.addLayout(self.input_form, 1, 0, 1, 1)
        layout.addLayout(self.learn_form, 1, 1, 1, 1)
        layout.addWidget(self.slider, 2, 0)
        layout.addWidget(self.start_stop_button, 2, 1)
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

        window_size = string.atoi(self.window_size_line_edit.text())
        m = string.atoi(self.m_line_edit.text())
        r = string.atoi(self.r_line_edit.text())
        hidden_layer_sizes = self.hidden_layer_sizes_line_edit.text().split(',')
        hidden_layer_sizes = [string.atoi(n) for n in hidden_layer_sizes]

        self.worker.setup(m=m, r=r, window_size=window_size, hidden_layer_sizes=hidden_layer_sizes, pretrain_step=1)
        self.updateWorker()
        self.worker.start()

    def stop(self):
        self.start_stop_button.setText('Start')
        self.start_stop_button.setEnabled(False)
        self.worker.stop()

    def updateWorker(self):
        self.worker.setDelay(self.getDelayValue())
        self.worker.setLearningParams(self.getLearningParams())

    def workerStarted(self):
        self.start_stop_button.setEnabled(True)
        self.window_size_line_edit.setReadOnly(True)
        self.m_line_edit.setReadOnly(True)
        self.r_line_edit.setReadOnly(True)
        self.hidden_layer_sizes_line_edit.setReadOnly(True)

    def updateFigure(self):
        # refresh canvas
        self.canvas.draw()

    def workerStopped(self):
        self.start_stop_button.setEnabled(True)
        self.window_size_line_edit.setReadOnly(False)
        self.m_line_edit.setReadOnly(False)
        self.r_line_edit.setReadOnly(False)
        self.hidden_layer_sizes_line_edit.setReadOnly(False)

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
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())