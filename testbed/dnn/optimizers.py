import numpy
import theano
import theano.tensor as tensor

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)

def sgd(lr, params, grads, x, mask, y, cost):
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
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(params, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, params, grads, x, mask, y, cost):
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

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
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


def rmsprop(lr, params, grads, x, mask, y, cost):
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

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
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

def my_rmsprop(lr, params, grads, x, mask, y, cost):
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
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]

    # build updates for g_list, r_list
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, decay_rate * rg + (1-decay_rate) * (g ** 2)) for rg, g in zip(running_grads, grads)]

    # build a function to update g_list and r_list
    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup,
                                    name='rmsprop_f_grad_shared')

    # build updates for params
    updir = [theano.shared(p.get_value() * numpy_floatX(0.)) for p in params]
    updir_new = [(ud, (lr*zg/tensor.sqrt(rg + epsilon))) for (ud, zg, rg) in zip(updir, zipped_grads, running_grads)]
    param_up = [(p, p - udn[1])
                for p, udn in zip(params, updir_new)]

    # build a function to update the model params
    f_update = theano.function([lr], [], updates=param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return (f_grad_shared, f_update)