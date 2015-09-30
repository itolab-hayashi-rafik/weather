import numpy
import pylab as plt
import re

LINE_DEFS = ['b.-', 'r.-', 'g.-']

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

        plt.show(block=False)

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

class LearningCurve(LineGraph):
    def __init__(self, costs_x, costs_y, fignum):
        self.costs_x = costs_x
        self.costs_y = costs_y

        super(LearningCurve, self).__init__(fignum)

    @property
    def xydata(self):
        return [
            (self.costs_x[0], self.costs_y[0]),
            (self.costs_x[1], self.costs_y[1]),
            (self.costs_x[2], self.costs_y[2]),
        ]

def parse_log(log_file='logs/1.log', unit='minibatch'):
    p_train_valid = r"epoch ([0-9]+), dataset ([0-9]+)/([0-9]+), minibatch ([0-9]+)/([0-9]+), took ([0-9.]+) secs, (train|validation) error ([0-9.]+)"
    p_test = r"(?:\s*)epoch ([0-9]+), dataset ([0-9]+)/([0-9]+), minibatch ([0-9]+)/([0-9]+), test error of best model ([0-9.]+)\(CrossE\), ([0-9.]+)\(MSE\)"

    r_train_valid = re.compile(p_train_valid)
    r_test = re.compile(p_test)

    train_costs = ([], []) # x, cost
    valid_costs = ([], []) # x, cost
    test_costs  = ([], []) # x, cost

    def get_index(epoch, dataset, n_datasets, minibatch, n_minibatches, unit=unit):
        assert unit in ('epoch', 'dataset', 'minibatch')
        if unit == 'epoch':
            return int(epoch)-1
        elif unit == 'dataset':
            return (int(epoch)-1)*int(n_datasets) + (int(dataset)-1)
        elif unit == 'minibatch':
            return (int(epoch)-1)*int(n_datasets) + (int(dataset)-1)*int(n_minibatches) + (int(minibatch)-1)


    with open(log_file) as f:
        line = f.readline()
        while line:
            m_train_valid = r_train_valid.match(line)
            m_test = r_test.match(line) if m_train_valid is None else None

            if m_train_valid is not None:
                # get groups
                groups = m_train_valid.groups()
                # 'epoch 3, dataset 21/31, minibatch 1/20, took 6.693754 secs, train error 7255.623047'
                # --> ('3', '21', '31', '1', '20', '6.693754', 'train', '7255.623047')
                index = get_index(groups[0], groups[1], groups[2], groups[3], groups[4])
                value = float(groups[7])

                if groups[6] == 'train':
                    train_costs[0].append(index)
                    train_costs[1].append(value)
                elif groups[6] == 'validation':
                    valid_costs[0].append(index)
                    valid_costs[1].append(value)
            elif m_test is not None:
                # get groups
                groups = m_test.groups()
                # '     epoch 1, dataset 1/31, minibatch 20/20, test error of best model 8352.040039(CrossE), 0.045281(MSE)'
                # --> ('1', '1', '31', '20', '20', '8352.040039', '0.045281')
                index = get_index(groups[0], groups[1], groups[2], groups[3], groups[4])
                value = float(groups[5])

                test_costs[0].append(index)
                test_costs[1].append(value)
            else:
                print('skipping line: {0}'.format(line)),

            line = f.readline()

    return (train_costs, valid_costs, test_costs)

if __name__ == '__main__':
    train_costs, valid_costs, test_costs = parse_log('logs/1.log')

    fig = LearningCurve(
        costs_x=(train_costs[0], valid_costs[0], test_costs[0]),
        costs_y=(train_costs[1], valid_costs[1], test_costs[1]),
        fignum=1
    )

    plt.show()