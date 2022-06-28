#!/usr/bin/env python3
#
#Andrew Geiss, May 2022
#
#Evaluates the various optics schemes (chebyshev interpolation, lookup tables, 
#and ANNs) on the randomly generated test data.

import numpy as np
from tqdm import tqdm
from keras.models import load_model
import keras.backend as K
from cam_aero_optics import modal_optics


def sci_notation(x):
    pow10 = np.log10(x)
    pow10 = np.floor(pow10)
    x = x*10**-pow10
    return x, int(pow10)

###############################################################################
#                 Evaluate the lookup tables on the test set                  #
###############################################################################
def linterp_weights(query,neighbors):
    n1,n2 = neighbors
    if query == n1:
        return 1,0
    elif query == n2:
        return 0,1
    assert query>n1 and query < n2
    d1 = abs(query-n1)
    d2 = abs(query-n2)
    w1 = (1-d1/(d1+d2))
    w2 = (1-d2/(d1+d2))
    return w1,w2

#a function to do the tri-linear interpolation:
def trinterp(query,axes,table):
    #get the hypercube around the query point
    idxs = []
    for a in range(3):
        idxs.append(np.where(axes[a] > query[a])[0][0]-1)
    cube = table[idxs[0]:idxs[0]+2,idxs[1]:idxs[1]+2,idxs[2]:idxs[2]+2]
    for a in range(3):
        w1,w2 = linterp_weights(query[a], axes[a][idxs[a]:idxs[a]+2])
        cube = cube[0,...]*w1 + cube[1,...]*w2
    return cube    


def compute_table_interp_errors():
    latex_table = []
    print('Computing Lookup Table Errors...')
    for table_name in ['9,17,257,33','33,33,513,65','65,65,1025,129','129,129,2049,257']:
        latex_table_row = []
        ax_sz = [int(sz) for sz in table_name.split(',')]
        ax_sz = np.delete(ax_sz,2)
        param_count = np.prod(ax_sz)*14*4*3 + np.prod(ax_sz)*16*4
        print('Table Dimensions: ' + table_name)
        print('Total Parameters: ' + '{:.1e}'.format(param_count))
        latex_table_row.append(param_count)
        for wvl_region in ['sw','lw']:
            print(wvl_region + ' table...')
            #test data columns: wavelength, irefr, irefi, rs, mode, abs, ext, asym
            test = np.load('./data/testing_data/' + wvl_region + '.npy')
            if wvl_region == 'sw':
                test_targets = test[:,-3:]
            else:
                test_targets = test[:,-3]
            test_inputs = test[:,:-3]
            
            #load the optics table:
            table = np.load('./data/optics_tables/' + wvl_region + '/' + table_name + '.npz')
            table_optics = table['optics']
            irefr, irefi = table['ref_index_real'], table['ref_index_imag']
            rs, wavelengths = table['surf_mode_radius'], table['wavelengths']
            
            #do absorption predictions:
            predictions = []
            for i in tqdm(range(test_targets.shape[0])):
                mode_num = int(test_inputs[i,-1])
                wvl_num = np.where(test_inputs[i,0] == wavelengths)[0][0]
                query = test_inputs[i,1:-1]
                axes = (irefr[wvl_num,:],irefi[wvl_num,:],rs)
                if wvl_region == 'sw':
                    pred = [trinterp(query,axes,table_optics[param,mode_num,wvl_num,...]) for param in range(3)]
                else:
                    pred = trinterp(query,axes,table_optics[0,mode_num,wvl_num,...])
                predictions.append(pred)
            predictions = np.array(predictions)
            np.save('./data/predictions/test/' + wvl_region + '_' + table_name + '.npy',predictions)
            errors = np.mean(np.abs(predictions-test_targets),axis=0)
            if wvl_region == 'sw':
                latex_table_row.extend(list(errors))
            else:
                latex_table_row.append(errors)
            print('errors: ' + str(errors) + '\n')
        latex_table.append(latex_table_row)
    print('\n\n\nGenerating latex table rows...')
    for row in latex_table:
        txt = 'Table: & $'
        for entry in row:
            x,p = sci_notation(entry)
            txt += str(np.round(x,1)) + '\\times 10^{' + str(p) + '}$ & $'
        txt = txt[:-3] + '\\\\'
        print(txt)
        






###############################################################################
#                   Evaluate the ANNs on the test data                        #
###############################################################################
def eval_anns():
    #the constants used here for input standardization are given in table A2 in the paper
    param_count = 0
    
    #test data columns: wavelength, irefr, irefi, rs, mode, abs, ext, asym
    test = np.load('./data/testing_data/sw.npy')
    sw_targets = test[:,-3:]
    inputs = [(np.log(test[:,0])+13.63357)/0.97014,
              (test[:,1]-1.63148)/0.18825,
              (np.log(test[:,2]+1E-6)+7.07294)/3.91970,
              (np.log(test[:,3]/test[:,0])+0.87509)/2.46624,
              (np.log(test[:,3])+14.50866)/2.26741,
              test[:,4] == 0, test[:,4] == 1,test[:,4] == 2,test[:,4] == 3]
    inputs = np.array(inputs).T
    ann = load_model('./data/anns/sw.ann')
    outputs = ann.predict(inputs,batch_size=2**15)
    sw_pred = outputs*np.array([2.2,4.6,1.0])[np.newaxis,:]
    param_count += np.sum([K.count_params(w) for w in ann.trainable_weights])
    np.save('./data/predictions/test/sw_ann.npy',sw_pred)
    
    #break the longwave data up based on the two sets of longwave bands:
    lwdata = np.load('./data/testing_data/lw.npy')
    lw_bands = np.load('./data/optics_tables/lw/33,33,513,65.npz')['wavelengths']
    lw2_bands = [lw_bands[i] for i in [0,1,6,7]]
    lwdata1,lwdata2 = [],[]
    for i in range(lwdata.shape[0]):
        if lwdata[i,0] in lw2_bands:
            lwdata2.append(lwdata[i,:])
        else:
            lwdata1.append(lwdata[i,:])
    lwdata1 = np.array(lwdata1)
    lwdata2 = np.array(lwdata2)
    
    #evaluate the first set of SW bands
    test = lwdata1[:,:-3]
    inputs = [(np.log(test[:,0])+11.84149)/0.53403,
              (test[:,1]-1.62030)/0.20075,
              (np.log(test[:,2]+1E-6)+7.07294)/3.91970,
              (np.log(test[:,3]/test[:,0])+2.66717)/2.32945,
              (np.log(test[:,3])+14.50866)/2.26741,
              test[:,4] == 0, test[:,4] == 1,test[:,4] == 2,test[:,4] == 3]
    inputs = np.array(inputs).T
    ann = load_model('./data/anns/lw1.ann')
    outputs1 = ann.predict(inputs,batch_size=2**15).squeeze()*2.2
    param_count += np.sum([K.count_params(w) for w in ann.trainable_weights])
    
    #evaluate the second set of longwave bands
    test = lwdata2[:,:-3]
    inputs = [(np.log(test[:,0])+10.34292)/1.64711,
              (test[:,1]-2.02558)/0.40344,
              (np.log(test[:,2]+1E-6)+6.99843)/3.92753,
              (np.log(test[:,3]/test[:,0])+4.16574)/2.80253,
              (np.log(test[:,3])+14.50866)/2.26741,
              test[:,4] == 0, test[:,4] == 1,test[:,4] == 2,test[:,4] == 3]
    inputs = np.array(inputs).T
    ann = load_model('./data/anns/lw2.ann')
    outputs2 = ann.predict(inputs,batch_size=2**15).squeeze()*2.2
    param_count += np.sum([K.count_params(w) for w in ann.trainable_weights])
    
    #re-combine the two sets of longwave bands:
    lw_targets = lwdata[:,-3]
    lw_pred = np.zeros(lw_targets.shape)
    i1, i2 = 0,0
    for i in range(lwdata.shape[0]):
        if lwdata[i,0] in lw2_bands:
            lw_pred[i] = outputs2[i2]
            i2 += 1
        else:
            lw_pred[i] = outputs1[i1]
            i1 += 1
    lw_pred = np.array(lw_pred)
    np.save('./data/predictions/test/lw_ann.npy',lw_pred)
    
    #generate the latex table:
    x,p = sci_notation(param_count)
    line = 'ANN: & $' + str(np.round(x,1)) + '\\times 10^{' + str(p) + '}$ & $'
    errors = list(np.mean(np.abs(sw_targets-sw_pred),axis=0))
    errors.append(np.mean(np.abs(lw_pred-lw_targets)))
    for e in errors:
        x,p = sci_notation(e)
        line += str(np.round(x,1)) + '\\times 10^{' + str(p) + '}$ & $'
    line = line[:-4] + ' \\\\'
    print(line)
    


###############################################################################
#        Evaluate the Ghan and Zaveri algorithm on the test data              #
###############################################################################
def compute_cheb_interp_errors():
    sw_bands = list(np.load('./data/optics_tables/sw/33,33,513,65.npz')['wavelengths'])
    swdata = np.load('./data/testing_data/sw.npy')
    targets = swdata[:,-3:]
    inputs = list(swdata[:,:-3])
    outputs = []
    
    #test data columns: wavelength, irefr, irefi, rs, mode, abs, ext, asym
    for sample in tqdm(inputs):
        wavelength, irefr, irefi, rs, mode = sample
        band = sw_bands.index(wavelength)
        pred = modal_optics(irefr, irefi, rs, int(mode), band, 'sw', mass_scaling=False)
        outputs.append(pred)
    outputs = np.array(outputs)
    np.save('./data/predictions/test/sw_cheb.npy',outputs)
    mae = list(np.mean(np.abs(outputs-targets),axis=0))
    
    lw_bands = list(np.load('./data/optics_tables/lw/33,33,513,65.npz')['wavelengths'])
    lwdata = np.load('./data/testing_data/lw.npy')
    targets = lwdata[:,-3]
    inputs = list(lwdata[:,:-3])
    outputs = []
    
    #test data columns: wavelength, irefr, irefi, rs, mode, abs, ext, asym
    for sample in tqdm(inputs):
        wavelength, irefr, irefi, rs, mode = sample
        band = lw_bands.index(wavelength)
        pred,_,_ = modal_optics(irefr, irefi, rs, int(mode), band, 'lw', mass_scaling=False)
        outputs.append(pred)
    outputs = np.array(outputs)
    np.save('./data/predictions/test/lw_cheb.npy',outputs)
    mae.append(np.mean(np.abs(outputs-targets)))
    
    #generate the latex table:
    param_count = 7*10*5*(14*3 + 16)*4
    x,p = sci_notation(param_count)
    line = 'EAMv1/CAM5: & $' + str(np.round(x,1)) + '\\times 10^{' + str(p) + '}$ & $'
    for e in mae:
        x,p = sci_notation(e)
        line += str(np.round(x,1)) + '\\times 10^{' + str(p) + '}$ & $'
    line = line[:-4] + ' \\\\'
    print(line)

if __name__ == '__main__':
    compute_table_interp_errors()
    eval_anns()
    compute_cheb_interp_errors()
