# -*- coding: utf-8 -*-
import os
import numpy
import theano
import dnn

def plot_graph(model, outdir):
    # check if the output directory exists and make directory if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print('building pretrain functions...'),
    f_pretrain = model.build_pretrain_function()
    print('done')

    print('building finetune functions...'),
    f_shared_grad, f_update = model.build_finetune_function()
    print('done')

    print('building predict function...'),
    f_predict = model.build_prediction_function()
    print('done')

    print('drawing graphs...')
    if f_pretrain is not None:
        theano.printing.pydotprint(f_pretrain, outfile="{0}/f_pretrain.png".format(outdir), var_with_name_simple=True)
    if f_shared_grad is not None:
        theano.printing.pydotprint(f_shared_grad, outfile="{0}/f_shared_grad.png".format(outdir), var_with_name_simple=True)
    if f_update is not None:
        theano.printing.pydotprint(f_update, outfile="{0}/f_update.png".format(outdir), var_with_name_simple=True)
    if f_predict is not None:
        theano.printing.pydotprint(f_predict, outfile="{0}/f_predict.png".format(outdir), var_with_name_simple=True)
    print('done')

if __name__ == '__main__':
    print('building model...'),
    numpy_rng = numpy.random.RandomState(89677)
    model = dnn.EncoderDecoderConvLSTM(numpy_rng, 5, 1, 28, 28, 5, [(3,1,3,3), (3,3,5,5)])
    print('done')

    plot_graph(model, 'graphs')
