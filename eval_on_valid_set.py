#!/usr/bin/env python3
#
#Andrew Geiss, May 2022
#
#Evaluates the various optics ANNs on the validation data

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from glob import glob
from tensorflow.keras.models import load_model
import numpy as np
import gc


def validate_random_anns(wvl):
    inputs = np.load('./data/training_data/' + wvl + '_inputs_test.npy')
    targets = np.load('./data/training_data/' + wvl + '_targets_test.npy')
    if not wvl == 'sw':
        targets = targets[:,np.newaxis]
    
    anns = glob('./data/anns/random_' + wvl + '/*.ann')
    anns.sort()
    params, names, errors = [], [], []
    for ann_file in anns:
        name = ann_file.split('/')[-1].split('.')[0]
        print('Evaluating ann ' + str(anns.index(ann_file)) + ': ' + name)
        model = load_model(ann_file)
        outputs = model.predict(inputs,verbose=True,batch_size=10_000)
        error = np.mean(np.abs(outputs-targets))
        print('Error: ' + str(error))
        errors.append(error)
        params.append(model.count_params())
        names.append(name)
    np.savez('./data/predictions/validation/' + wvl + '_validation',name = names,params = params,error = errors)

def validate_benchmark_anns(wvl):
    inputs = np.load('./data/training_data/' + wvl + '_inputs_test.npy')
    targets = np.load('./data/training_data/' + wvl + '_targets_test.npy')
    sizes = np.array([500,1000,2500,5000,7500,10_000,15_000,25_000,50_000,100_000])
    layers = [1,2,3,4,5,6]
    
    layer_dim = []
    for l in layers:
        size_dim = []
        for s in sizes:
            ann_files = glob('./data/anns/benchmark_' + wvl + '/P' + str(s) + '_L' + str(l) + '*.ann')
            errors = np.nan*np.ones(shape=(5,))
            for i_file in range(min(len(ann_files),5)):
                fname = ann_files[i_file]
                print(fname)
                ann = load_model(ann_files[i_file])
                outputs = ann.predict(inputs,batch_size=100_000)
                if outputs.shape[1]>1 and wvl != 'sw':
                    outputs = outputs[:,1]
                errors[i_file] = np.mean(np.abs(outputs.squeeze()-targets))
                del ann
                gc.collect()
            size_dim.append(errors)
        layer_dim.append(size_dim)
    
    errors = np.array(layer_dim)
    np.save('./predictions/validation/bench_valid_errors_' + wvl + '.npy',errors)


if __name__ == '__main__':
    for wvl in ['sw','lw1','lw2']:
        validate_random_anns(wvl)
        validate_benchmark_anns(wvl)
