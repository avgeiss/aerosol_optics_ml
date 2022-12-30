#!/usr/bin/env python3
#
#Andrew Geiss, May 2022
#
#Evaluates the various optics ANNs on the validation data

from glob import glob
from tensorflow.keras.models import load_model
import numpy as np
import gc

def validate_random_anns(wvl):
    inputs = np.double(np.load('./data/validation/' + wvl + '_inputs.npy'))
    targets = np.double(np.load('./data/validation/' + wvl + '_targets.npy'))
    anns = sorted(glob('./data/anns/' + wvl + '/random/*'))
    params, names, errors = [], [], []
    for ann_file in anns:
        names.append(ann_file.split('/')[-1].split('.')[0])
        model = load_model(ann_file)
        outputs = model.predict(inputs,batch_size=2**16,verbose=False).squeeze()
        errors.append(np.mean(np.abs(outputs-targets)))
        params.append(model.count_params())
        print('Evaluating ann ' + ann_file.split('/')[-1] + ': ' + str(errors[-1]))
        del model
        gc.collect()
    np.savez('./data/evaluation/validation_' + wvl + '_random.npz',name = names,params = params,error = errors)

def validate_benchmark_anns(wvl):
    inputs = np.double(np.load('./data/validation/' + wvl + '_inputs.npy'))
    targets = np.double(np.load('./data/validation/' + wvl + '_targets.npy'))
    params = np.array([500,1000,2500,5000,7500,10_000,15_000,20_000,50_000,100_000])
    layers = [2,3,4,5,6]
    errors = np.zeros([5,10,5])
    for ilayers in range(len(layers)):
        for iparams in range(len(params)):
            files = glob('./data/anns/' + wvl + '/benchmark/L' + str(layers[ilayers]) + '_P' + str(params[iparams]) + '_*')
            for ifiles in range(len(files)):
                print(files[ifiles])
                ann = load_model(files[ifiles])
                outputs = ann.predict(inputs,batch_size=2**16).squeeze()
                errors[ilayers,iparams,ifiles] = np.mean(np.abs(outputs-targets))
                del ann
                gc.collect()
    np.save('./data/evaluation/validation_' + wvl + '_benchmark.npy',errors)

validate_benchmark_anns('sw')
validate_benchmark_anns('lw')
validate_random_anns('sw')
validate_random_anns('lw')