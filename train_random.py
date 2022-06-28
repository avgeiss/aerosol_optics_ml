#!/usr/bin/env python3
#
#Andrew Geiss, Mar 2022
#
#This script trains a single randomly generated ann architecture. The appropriate
#wavelength region should be set below:
wvl_region='sw' #choose from 'sw' 'lw1' 'lw2'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from neural_networks import RandomANN
import time
from tensorflow.keras import backend as K

print('Creating Random ANN...',flush=True)
if wvl_region[:2] == 'lw':
    ann = RandomANN(nio = [9,1])
else:
    ann = RandomANN(nio = [9,3])
ann.build()

print('Loading Data...',flush=True)
inputs = np.load('./data/training_data/' + wvl_region + '_inputs_train.npy')
targets = np.load('./data/training_data/' + wvl_region + '_targets_train.npy')

print('Training...',flush=True)
esize=10_000
bsize=32
N = inputs.shape[0]
lr_scheme = [8_000,5_000,7_000] #([total epochs, reduce lr 1/10 #1, reduce lr 1/10 #2])
if wvl_region == 'lw2':
    lr_scheme = [4_000,2_500,3_500]

for epoch in range(lr_scheme[0]):
    t0 = time.time()
    binds = np.random.randint(N,size=(bsize*esize,))
    hist = ann.train(inputs[binds,...],targets[binds,...],batch_size=bsize)
    delta = time.time()-t0
    if len(ann.loss)%1000==0:
        ann.save('./data/anns/random_' + wvl_region + '/')
        ann.ann.save('./data/anns/random_' + wvl_region + '/' + ann.name + '.ann')
    if len(ann.loss)%1==0:
        print('Training Updates:' + str(len(ann.loss)*esize).rjust(10) + ' Loss: ' + str(ann.loss[-1]) + '   Duration = ' + str(delta),flush=True)
    if epoch == lr_scheme[1]:
        print('Lowering learning rate')
        K.set_value(ann.ann.optimizer.learning_rate,0.0001)
    elif epoch == lr_scheme[2]:
        print('Lowering learning rate')
        K.set_value(ann.ann.optimizer.learning_rate,0.00001)
    
ann.save('./data/anns/random_' + wvl_region + '/')
ann.ann.save('./data/anns/random_' + wvl_region + '/' + ann.name + '.ann')