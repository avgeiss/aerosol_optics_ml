#!/usr/bin/env python3
#
#Andrew Geiss, May 2022
#
#performs a final round of training on select ANN architectures. Uses both the
#training and validation datasets and trains for twice as long.


#name = 'b83b12a6a0cd45dda2149bab6103542b';wvl_region='lw1';nio=[9,1]
#name = '92c7de5a466f4bab83e49b468da295c1';wvl_region='lw2';nio=[9,1]
name = '77b3cf4eda3748db9c475a474d2183da';wvl_region='sw';nio=[9,3]

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from neural_networks import RandomANN
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

print('Loading Data...',flush=True)
inputs = np.load('./data/training_data/' + wvl_region + '_inputs_train.npy')
targets = np.load('./data/training_data/' + wvl_region + '_targets_train.npy')
inputs2 = np.load('./data/training_data/' + wvl_region + '_inputs_test.npy')
targets2 = np.load('./data/training_data/' + wvl_region + '_targets_test.npy')
inputs = np.concatenate((inputs,inputs2),axis=0)
targets = np.concatenate((targets,targets2),axis=0)

print('Initializing ANN...',flush=True)
ann = RandomANN(nio  = nio, load_fname='./data/anns/random_' + wvl_region + '/' + name + '.npz')
ann.build()
ann.ann.compile(optimizer=Adam(learning_rate=0.001),loss='MSE')
ann.loss = []

print('Training...',flush=True)
esize=10_000
bsize=32
N = inputs.shape[0]
lr_scheme = [16_000,10_000,14_000] #([total epochs, reduce lr 1/10 #1, reduce lr 1/10 #2])
if wvl_region == 'lw2':
    lr_scheme = [8_000,5_000,7_000]

for epoch in range(lr_scheme[0]):
    t0 = time.time()
    binds = np.random.randint(N,size=(bsize*esize,))
    hist = ann.train(inputs[binds,...],targets[binds,...],batch_size=bsize)
    delta = time.time()-t0
    #if len(ann.loss)%1000==0:
    #    ann.ann.save('./data/anns/checkpoints/' + wvl_region + '_' + str(epoch).zfill(5) + '.ann')
    if len(ann.loss)%1==0:
        print('Training Updates:' + str(len(ann.loss)*esize).rjust(10) + ' Loss: ' + str(ann.loss[-1]) + '   Duration = ' + str(delta),flush=True)
    if epoch == lr_scheme[1]:
        print('Lowering learning rate')
        K.set_value(ann.ann.optimizer.learning_rate,0.0001)
    elif epoch == lr_scheme[2]:
        print('Lowering learning rate')
        K.set_value(ann.ann.optimizer.learning_rate,0.00001)
    
ann.ann.save('./data/anns/' + wvl_region + '.ann')
