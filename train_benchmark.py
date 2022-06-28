#!/usr/bin/env python3
#
#Andrew Geiss, April 2022
#
#this script trains some simple FF-MLP ANN architectures as benchmarks for
#evaluating the random ANNs against

#parse command line inputs:
import sys
args = sys.argv
if len(args) > 1:
    wvl_region, params, layers = args[1], int(args[2]), int(args[3])
else:
    wvl_region='sw'; params = 10_000; layers = 6
n_inputs = 9
if wvl_region == 'sw':
    n_outputs = 3
else:
    n_outputs = 1

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from neural_networks import benchmark_mlp
import numpy as np
import time
from tensorflow.keras import backend as K
from uuid import uuid4

#build the ANN
print('Building benchmark ANN:')
ann, params_actual = benchmark_mlp(layers,params,n_inputs,n_outputs)
print('params requested = ' + str(params) + '\nparams actual = ' + str(params_actual))

print('Loading Data',flush=True)
inputs = np.load('./data/training_data/' + wvl_region + '_inputs_train.npy')
targets = np.load('./data/training_data/' + wvl_region + '_targets_train.npy')

print('Training...',flush=True)
esize=10_000
bsize=32
annid = uuid4().hex
name = 'P' + str(params) + '_L' + str(layers) + '_' + annid + '.ann'
loss = []
N = inputs.shape[0]
for epoch in range(8_000):
    t0 = time.time()
    binds = np.random.randint(N,size=(bsize*esize,))
    hist = ann.fit(inputs[binds,...],targets[binds,...],batch_size=bsize,verbose=0)
    loss.append(hist.history['loss'])
    delta = time.time()-t0
    #if len(loss)%100==0:
    #    ann.save('./data/anns/benchmark_' + wvl_region + '/' + name + '.ann')
    #    np.save('./data/anns/benchmark_' + wvl_region + '/'  + name + '_loss.npy',loss)
    if epoch == 5_000:
        print('Lowering learning rate',flush=True)
        K.set_value(ann.optimizer.learning_rate,0.0001)
    elif epoch == 7_000:
        print('Lowering learning rate',flush=True)
        K.set_value(ann.optimizer.learning_rate,0.00001)
    print('Training Updates:' + str(len(loss)*esize).rjust(10) + ' Loss: ' + str(loss[-1]) + '   Duration = ' + str(delta),flush=True)

ann.save('./data/anns/benchmark_' + wvl_region + '/' + name + '.ann')
np.save('./data/anns/benchmark_' + wvl_region + '/'  + name + '_loss.npy',loss)
