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

def parse_log(log_file, unit='minibatch'):
    p_train = r"Epoch ([0-9]+)/([0-9]+), Update ([0-9]+)/([0-9]+), took ([0-9.]+) secs, Cost: ([0-9.]+)"
    p_valid_test = r"(?:\s*)\(validtion\) Epoch ([0-9]+)/([0-9]+), Update ([0-9]+)/([0-9]+), Train: ([0-9.]+), Valid: ([0-9.]+), Test: ([0-9.]+)"

    r_train = re.compile(p_train)
    r_valid_test = re.compile(p_valid_test)

    train_costs = ([], []) # x, cost
    valid_costs = ([], []) # x, cost
    test_costs  = ([], []) # x, cost

    def get_index(epoch, minibatch, n_minibatches, unit=unit):
        assert unit in ('epoch', 'minibatch')
        (epoch, minibatch, n_minibatches) = \
            (int(epoch), int(minibatch), int(n_minibatches))
        if unit == 'epoch':
            return (epoch-1)
        elif unit == 'minibatch':
            return (epoch-1)*n_minibatches + (minibatch-1)


    with open(log_file) as f:
        line = f.readline()
        while line:
            m_train_valid = r_train.match(line)
            m_test = r_valid_test.match(line) if m_train_valid is None else None

            if m_train_valid is not None:
                # get groups
                groups = m_train_valid.groups()
                # 'Epoch 32/5000, Update 624/625, took 6.02633500099 secs, Cost: 3698.87304688'
                # --> ('32', '5000', '624', '625', '6.02633500099', '3698.87304688')
                index = get_index(groups[0], groups[2], groups[3])
                value = float(groups[5])

                train_costs[0].append(index)
                train_costs[1].append(value)
            elif m_test is not None:
                # get groups
                groups = m_test.groups()
                # ' (validtion) Epoch 32/5000, Update 624/625, Train: 4013.97666211, Valid: 4962.40923047, Test: 4918.42494234'
                # --> ('32', '5000', '624', '625', '4013.97666211', '4962.40923047', '4918.42494234')
                index = get_index(groups[0], groups[2], groups[3])
                train_value = float(groups[4])
                valid_value = float(groups[5])
                test_value = float(groups[6])

                valid_costs[0].append(index)
                valid_costs[1].append(valid_value)
                test_costs[0].append(index)
                test_costs[1].append(test_value)
            else:
                print('skipping line: {0}'.format(line)),

            line = f.readline()

    return (train_costs, valid_costs, test_costs)

if __name__ == '__main__':
    train_costs, valid_costs, test_costs = parse_log('logs/3.log', unit='minibatch')

    fig = LearningCurve(
        costs_x=(train_costs[0], valid_costs[0], test_costs[0]),
        costs_y=(train_costs[1], valid_costs[1], test_costs[1]),
        fignum=1
    )

    plt.show()