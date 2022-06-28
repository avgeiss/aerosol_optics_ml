#!/usr/bin/env python3
#
#Andrew Geiss, May 2022
#
#generates a test set for evaluating optics schemes by randomly selecting query
#points to evaluate optics from continuous distributions (instead of the regular
#grids used for the training and validation data)

import numpy as np
import optics_utils as utils
from multiprocess import Pool
import time

nrad = 2049
N_samples = 1_000_000
wvl_region = 'lw'
    
#get the wavelength bands:
if wvl_region=='sw':
    wvl_bands = utils.SW_BANDS
else:
    wvl_bands = utils.LW_BANDS
nbands = wvl_bands.shape[0]

#get the refractive index bounds
crefw = utils.read_water_refindex(wvl_region)
crefa = utils.read_modal_optics(wvl_region)

#get individual particle radii:
rmmin, rmmax = utils.rm_range
rmin, rmax = utils.r_range
radii = np.exp(np.linspace(np.log(rmin),np.log(rmax),nrad))

#determine the boundaries for the refractive index ranges
ibounds, rbounds = [],[]
for band in range(nbands):
    crefs = np.append(np.concatenate(crefa,axis=0)[:,band],crefw[band])
    ibounds.append([np.min(np.imag(crefs)), np.max(np.imag(crefs))])
    rbounds.append([np.min(np.real(crefs)), np.max(np.real(crefs))])

def random_params():
    band = np.random.randint(nbands)
    wavelength = np.mean(wvl_bands[band,:])
    irefr = np.random.uniform(*rbounds[band])
    irefi = ibounds[band][1]*np.exp(np.random.uniform(np.log(1E-6),0))
    rs = np.exp(np.random.uniform(np.log(rmmin),np.log(rmmax)))
    return wavelength, irefr, irefi, rs

def bulk_optics(wavelength,refr,refi,rs):
    particle_optics = np.array([utils.mie(r,wavelength,refr,refi) for r in radii])
    qabs,qsca,g = [particle_optics[:,i] for i in range(3)]
    modal_abs, modal_ext, modal_asm = utils.modal_int_lognorm(qabs,qsca,g,radii[np.newaxis,np.newaxis,:],[rs])
    return modal_abs, modal_ext, modal_asm
 
#do mie calculations
nsteps = 100
inputs,optics = [],[]
p = Pool(24)
for i in range(nsteps):
    start_time = time.time()
    cur_inputs = [random_params() for _ in range(N_samples//nsteps)]
    optics.append(np.array(p.starmap(bulk_optics,cur_inputs)))
    inputs.append(cur_inputs)
    duration = (time.time()-start_time)/(60*24)
    print(str(i) + '/' + str(nsteps) + '   Time Remaining: ' + str(np.round(duration*(nsteps-i-1),1)) + 'hrs',flush=True)
    
p.close()
inputs = np.concatenate(inputs,axis=0)
optics = np.concatenate(optics,axis=0)

#package and save data
inputs = np.array(inputs)
optics = np.array(optics)
rand_mode = np.random.randint(0,4,size=(optics.shape[0],1))
optics = np.array([optics[i,:,rand_mode[i],0].squeeze() for i in range(optics.shape[0])])
data = np.concatenate((inputs,rand_mode,optics),axis=1)

#the data is saved as one matrix with columns:
#wavelength, real refractive index, imaginary refractive index, mode radius, 
#mode number, absorption efficiency, extinction efficiency, assymetry parameter
np.save('./data/testing_data/' + wvl_region + '.npy',data)
