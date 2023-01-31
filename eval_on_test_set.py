#!/usr/bin/env python3
#
#Andrew Geiss, May 2022
#
#Evaluates the various optics schemes (chebyshev interpolation, lookup tables, 
#and ANNs) on the test data.

import numpy as np
from tqdm import tqdm
from keras.models import load_model
from cam_aero_optics import modal_optics
from generate_train_set import standardize, one_hot
from numba import njit
ANN_NAMES = {'sw': '2272914092e14e71ba8b3398f9bdfd3c',
             'lw': 'eae77c85f3e54c09a836ff613d546c6f'}
BENCH_NAMES = {'sw': 'L3_P20000_e759301ebf074facb4eafd45b7bee4e3',
               'lw': 'L3_P5000_e149d701e4a34b09af5debebcf4c3c14'}

#computes the scattering and SSA
def compute_scattering(x):
    #ensure absorption does not exceed extinction
    swap = x[:,0]>x[:,1]
    x[swap,1] = x[swap,0]
    #compute scattering
    qs = x[:,1]-x[:,0]
    #compute SSA
    ssa = qs/x[:,1]
    ssa[x[:,1]<0.01] = np.nan
    ssa = np.clip(ssa,0,1)
    return np.concatenate([x,qs[:,np.newaxis],ssa[:,np.newaxis]],axis=-1)

#load the test dataset:
def load_testing_data(wvl_region):
    nax = np.newaxis
    # dataset = np.load('./data/optics_tables/' + wvl_region + '/16,16,513,32_test_set.npz')#for fast testing
    dataset = np.load('./data/optics_tables/' + wvl_region + '/128,128,2049,256_test_set.npz')
    optics = dataset['optics']
    shape = optics.shape[1:]
    if wvl_region == 'sw':
        optics = optics.transpose((1,2,3,4,5,0)).reshape((np.prod(shape),3))
        optics = compute_scattering(optics)
    else:
        optics = optics[0,...].flatten()
    inputs = np.zeros(shape + (5,),dtype='float32')
    inputs[...,0] = np.broadcast_to(dataset['mode'][:,nax,nax,nax,nax],shape)
    inputs[...,1] = np.broadcast_to(dataset['wavelengths'][nax,:,nax,nax,nax],shape)
    inputs[...,2] = np.broadcast_to(dataset['ref_index_real'][nax,:,:,nax,nax],shape)
    inputs[...,3] = np.broadcast_to(dataset['ref_index_imag'][nax,:,nax,:,nax],shape)
    inputs[...,4] = np.broadcast_to(dataset['surf_mode_radius'][nax,nax,nax,nax,:],shape)
    inputs = inputs.reshape((np.prod(shape),5))
    return inputs, optics

def compute_ground_truth(wvl_region):
    _, optics = load_testing_data(wvl_region)
    np.save('./data/evaluation/test_' + wvl_region + '_mie.npy',optics)

def compute_cheb_interp_errors(wvl_region):
    print('Testing ' + wvl_region + ' EAM scheme...')
    band_wvls = list(np.float32(np.load('./data/optics_tables/' + wvl_region + '/33,33,513,65.npz')['wavelengths']))
    inputs,optics = load_testing_data(wvl_region)
    outputs = np.zeros((optics.shape[0],3),dtype='float32')
    for i in tqdm(range(inputs.shape[0])):
        mode,wvl,refr,refi,rs = inputs[i,:]
        outputs[i,:] = modal_optics(refr, refi, rs, int(mode), band_wvls.index(wvl), wvl_region, mass_scaling=False)
    if wvl_region=='sw':
        outputs = compute_scattering(outputs)
    else:
        outputs = outputs[:,0]
    np.save('./data/evaluation/test_' + wvl_region + '_cheb.npy',outputs)

def compute_ann_errors(wvl_region, benchmark=False):
    print('Testing ' + wvl_region + ' ann...')
    inputs,_ = load_testing_data(wvl_region)
    mode, wvl, refr, refi, rs = [inputs[:,i] for i in range(5)]
    refi += 1E-6
    rssp = rs/wvl
    wvl, refr, refi, rssp, rs = [standardize(data, varname, wvl_region) for data, varname in 
                                 zip([wvl,refr,refi,rssp,rs], ['wavelength','refr','refi','rssp','rs'])]
    mode = one_hot(mode)
    inputs = np.stack([inp_var for inp_var in [wvl, refr, refi, rssp, rs]],axis=1)
    inputs = np.concatenate([inputs,mode],axis=1)
    if benchmark:
        ann = load_model('./data/anns/' + wvl_region + '/benchmark/' + BENCH_NAMES[wvl_region])
    else:
        ann = load_model('./data/anns/' + wvl_region + '/random/' + ANN_NAMES[wvl_region])
    outputs = ann.predict(inputs,batch_size=2**14)
    if wvl_region == 'sw':
        scales = np.array([1.6,3.4,1.0])
        outputs *= scales[np.newaxis,:]
        outputs = compute_scattering(outputs)
    else:
        outputs = 1.6*outputs.squeeze()
    if benchmark:
        np.save('./data/evaluation/test_' + wvl_region + '_benchmark_ann.npy',outputs)
    else:
        np.save('./data/evaluation/test_' + wvl_region + '_ann.npy',outputs)

@njit
def linterp_weights(query,rng):
    assert query > rng[0] and query < rng[-1], 'query point outside of range'
    i = 0
    while rng[i+1] < query:
        i += 1
    w = (rng[i+1]-query)/(rng[i+1]-rng[i])
    return i, (w,1-w)

@njit
def trinterp(cube,w1,w2,w3):
    cube = cube[:,0,:,:]*w1[0] + cube[:,1,:,:]*w1[1]
    cube = cube[:,0,:]*w2[0] + cube[:,1,:]*w2[1]
    cube = cube[:,0]*w3[0] + cube[:,1]*w3[1]
    return cube

def compute_table_errors(wvl_region, table_res):
    print('Testing ' + wvl_region + ' ' + table_res + ' optics table...')
    test_inputs, _ = load_testing_data(wvl_region)
    N = test_inputs.shape[0]
    table_dataset = np.load('./data/optics_tables/' + wvl_region + '/' + table_res + '.npz')
    table_optics = table_dataset['optics']
    wvls, refrs, refis, rss = [table_dataset[var_name] for var_name in 
                                      ['wavelengths','ref_index_real','ref_index_imag','surf_mode_radius']]
    wvl_inds = np.int16(np.argmin(np.abs(test_inputs[:,1:2] - wvls[np.newaxis,:]),axis=-1))
    interped_optics = np.zeros((N,3),dtype='float32')
    for i in tqdm(range(N)):
        iw = wvl_inds[i]
        im = int(test_inputs[i,0])
        refr_idx, refr_w = linterp_weights(test_inputs[i,2],refrs[iw,:])
        refi_idx, refi_w = linterp_weights(test_inputs[i,3],refis[iw,:])
        rs_idx,   rs_w   = linterp_weights(test_inputs[i,4],rss)
        interped_optics[i,:] = trinterp(table_optics[:,im,iw,refr_idx:refr_idx+2,refi_idx:refi_idx+2,rs_idx:rs_idx+2],
                                        refr_w, refi_w, rs_w)
    if wvl_region == 'sw':
        interped_optics = compute_scattering(interped_optics)
    else:
        interped_optics = interped_optics[:,0]
    np.save('./data/evaluation/test_' + wvl_region + '_' + table_res + '.npy',interped_optics)
    
def compute_error_table():
    sw_truth = np.double(np.load('./data/evaluation/test_sw_mie.npy'))
    lw_truth = np.double(np.load('./data/evaluation/test_lw_mie.npy'))
    errors = np.zeros((7,6))
    model_names = ['ann','benchmark_ann','cheb','9,17,257,33','33,33,513,65','65,65,1025,129','129,129,2049,257']
    for i in range(errors.shape[0]):
        errors[i,:-1] = np.nanmean(np.abs(sw_truth-np.load('./data/evaluation/test_sw_' + model_names[i] + '.npy')),axis=0)
        errors[i,-1] = np.nanmean(np.abs(lw_truth-np.load('./data/evaluation/test_lw_' + model_names[i] + '.npy')))
    var_names = ['qabs','qext','g','qsca','SSA','qabs_lw']
    np.savez('./data/evaluation/error_table.npz',errors = errors, columns = var_names, rows = model_names)
    
def write_latex_table():
    #generates the MAE table from the paper in latex notation
    
    def sci_notation(x):
        pow10 = np.log10(x)
        pow10 = np.floor(pow10)
        x = x*10**-pow10
        return x, int(pow10)
    
    errors = np.load('./data/evaluation/error_table.npz')['errors']
    lines = ['\\begin{center}','\\begin{tabular}{ccccccccc}','\hline', 
              'Method & N-Params. & $\overline{Q}_{\\text{Abs.}}$ (SW) & $\overline{Q}_{\\text{Ext.}}$ ' + 
              '(SW) & $\overline{g}$ (SW) & $\overline{Q}_{\\text{Sca.}}$ (SW) & $\overline{SSA}$ (SW) ' + 
              '& $\overline{Q}_{\\text{Abs.}}$ (LW)\\\\',
              '\hline']
    model_names = ['Random ANN', 'Serial ANN', '\cite{Ghan_2007}'] + ['Lookup Table']*4
    params = [4] + list(range(4,10))
    for i in range(len(model_names)):
        line = model_names[i] + ': & $10^{' + str(params[i]) + '}$ '
        for j in range(errors.shape[1]):
            x, pow10 = sci_notation(errors[i,j])
            line += '& $' + str(np.round(x,1)) + ' \\times 10^{' + str(pow10) + '}$ '
        line += '\\\\'
        lines.append(line)
        if i == 1:
            lines.append('\hline')
    lines += ['\hline','\end{tabular}','\end{center}','\end{table}']
    
    file = open('./figures/table.txt','w')
    for line in lines:
        file.write(line + '\n')
    file.close()

# compute_ground_truth('sw')
# compute_ground_truth('lw')
# compute_cheb_interp_errors('sw')
# compute_cheb_interp_errors('lw')
# compute_ann_errors('sw')
# compute_ann_errors('lw')
# compute_table_errors('sw', '9,17,257,33')
# compute_table_errors('sw', '33,33,513,65')
# compute_table_errors('sw', '65,65,1025,129')
# compute_table_errors('sw', '129,129,2049,257')
# compute_table_errors('lw', '9,17,257,33')
# compute_table_errors('lw', '33,33,513,65')
# compute_table_errors('lw', '65,65,1025,129')
# compute_table_errors('lw', '129,129,2049,257')
# compute_ann_errors('sw',benchmark=True)
# compute_ann_errors('lw',benchmark=True)
# compute_error_table()
# errors = write_latex_table()