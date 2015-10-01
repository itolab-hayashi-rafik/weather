import numpy
import theano
import theano.tensor as tensor

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def sgd(lr, params, grads, x, mask, y, shared_x, shared_mask, shared_y, index, batch_size, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0.) for p in params]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([index], cost, updates=gsup,
                                    givens={
                                        x: shared_x[index * batch_size: (index + 1) * batch_size],
                                        mask: shared_mask[index * batch_size: (index + 1) * batch_size],
                                        y: shared_y[index * batch_size: (index + 1) * batch_size]
                                    },
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(params, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, params, grads, x, mask, y, shared_x, shared_mask, shared_y, index, batch_size, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    pramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([index], cost, updates=zgup + rg2up,
                                    givens={
                                               x: shared_x[index * batch_size: (index + 1) * batch_size],
                                               mask: shared_mask[index * batch_size: (index + 1) * batch_size],
                                               y: shared_y[index * batch_size: (index + 1) * batch_size]
                                    },
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, params, grads, x, mask, y, shared_x, shared_mask, shared_y, index, batch_size, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    pramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([index], cost,
                                    updates=zgup + rgup + rg2up,
                                    givens={
                                        x: shared_x[index * batch_size: (index + 1) * batch_size],
                                        mask: shared_mask[index * batch_size: (index + 1) * batch_size],
                                        y: shared_y[index * batch_size: (index + 1) * batch_size]
                                    },
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(params, updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def my_rmsprop(lr, params, grads, x, mask, y, shared_x, shared_mask, shared_y, index, batch_size, cost):
    '''
    An implementation of RMSProp
    :param lr:
    :param params:
    :param grads:
    :param x:
    :param mask:
    :param y:
    :param cost:
    :return:
    '''
    decay_rate = 0.9
    epsilon = 1E-6

    # initialize running grads
    r_list = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]

    # build update lists
    new_r_list = [decay_rate*r + (1-decay_rate)*tensor.square(g) for (g, r) in zip(grads, r_list)]
    upd_r_list = [(r, new_r) for (r, new_r) in zip(r_list, new_r_list)]
    new_w_list = [w - (lr / (tensor.sqrt(new_r)+epsilon)) * g for (g, w, new_r) in zip(grads, w_list, new_r_list)]
    upd_w_list = [(w, new_w) for (w, new_w) in zip (params, new_w_list)]

    # build a function to calculate r
    f_grad_shared = theano.function([index], cost,
                                    updates=upd_r_list,
                                    givens={
                                        x: shared_x[index * batch_size: (index + 1) * batch_size],
                                        mask: shared_mask[index * batch_size: (index + 1) * batch_size],
                                        y: shared_y[index * batch_size: (index + 1) * batch_size]
                                    },
                                    name='rmsprop_f_grad_shared')

    # build a function to calculate and update params w
    f_update = theano.function([lr], [],
                               updates=upd_w_list,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return (f_grad_shared, f_update)