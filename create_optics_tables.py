#!/usr/bin/env python3
#
#Andrew V. Geiss, Feb 8th 2022
#
#this file contains subroutines that generate tables of aerosol optical properties
#integrated over a range of size distributions with the table bounds decided
#by the wavelenght bands and aerosol modes used by RRTMG and MAM in E3SM

import numpy as np
import optics_utils as utils
import concurrent

#Flexible parallelization of Mie calculations
def looped_mie(inputs):
    #the function called by each thread that loops over its vectorized inputs:
    N = inputs.shape[0]
    opt_params = np.zeros((N,3),dtype='float32')
    for i in range(N):
        opt_params[i,:] = utils.mie(*inputs[i,:])
    return opt_params
    
def parallel_mie(radii, wavelength, refr, refi):
    
    #vectorize inputs:
    refr,refi,radii = np.meshgrid(refr,refi,radii,indexing='ij')
    shape = refr.shape
    inputs = np.stack([dat.flatten() for dat in [radii, wavelength*np.ones(shape), refr, refi]], axis=1)
    
    #break inputs into chunks for each thread to work on:
    N = inputs.shape[0]
    CL = int(np.ceil(N/(utils.n_workers*10)))
    inputs = [inputs[i:i+CL,:] for i in range(0,N,CL)]
    threader = concurrent.futures.ProcessPoolExecutor(max_workers=utils.n_workers)
    threads = [threader.submit(looped_mie,inp) for inp in inputs]
    mie_effs = [thr.result() for thr in threads]
    mie_effs = np.concatenate(mie_effs)
    threader.shutdown()
    
    #reshape and return the results:
    qabs, qsca, g  = [np.reshape(mie_effs[:,i],shape) for i in range(3)]
    return qabs, qsca, g

def ref_ind_range(spect_region,nrefr,nrefi,band):
    #load the optical properties of the aerosols and water:
    crefw = utils.read_water_refindex(spect_region)
    crefa = utils.read_modal_optics(spect_region)
    
    #determine the boundaries for the refractive index ranges
    crefs = np.append(np.concatenate(crefa,axis=0)[:,band],crefw[band])
    ibounds = [np.min(np.imag(crefs)), np.max(np.imag(crefs))]
    rbounds = [np.min(np.real(crefs)), np.max(np.real(crefs))]
    
    #define the grid
    refi = ibounds[1]*np.exp(np.linspace(np.log(1E-6),0,nrefi))
    refi[0] = 0
    refr = np.linspace(*rbounds,nrefr)
    return refr, refi

def modal_optics_table(nrefr=128,nrefi=128,nrad=2048,ndist=256,test_dataset=False):
    #file name suffix
    suffix = str(nrefr) + ',' + str(nrefi) + ',' + str(nrad) + ',' + str(ndist)
    if test_dataset:
        suffix = str((nrefr-1)//2) + ',' + str((nrefi-1)//2) + ',' + str(nrad) + ',' + str((ndist-1)//2) + '_test_set'
    rmmin, rmmax = utils.rm_range
    rs = np.exp(np.linspace(np.log(rmmin),np.log(rmmax),ndist))
    if test_dataset:
        rs = rs[1::2]
    
    for wl_region, wl_bands in zip(['sw','lw'],[utils.SW_BANDS, utils.LW_BANDS]):
        optics_table = []
        wavelengths = np.mean(wl_bands,axis=1)
        refrs, refis = [], []
        
        for band in range(len(wl_bands)):
            print('processing ' + wl_region + ' ' + str(int(np.round(wavelengths[band]*1E9))) + 'nm band')
            
            #compute particle optical properties:
            rmin, rmax = utils.r_range
            radii = np.exp(np.linspace(np.log(rmin),np.log(rmax),nrad))
            refr, refi = ref_ind_range(wl_region,nrefr,nrefi,band)
            if test_dataset:
                refr = refr[1::2]
                refi = refi[1::2]
            qabs, qsca, g = parallel_mie(radii, wavelengths[band], refr, refi)
        
            #integrate over size distributions:
            modal_abs, modal_ext, modal_asm = utils.modal_int_lognorm(qabs,qsca,g,radii[np.newaxis,np.newaxis,:],rs)
            optics_table.append([modal_abs, modal_ext, modal_asm])
            refrs.append(refr);refis.append(refi)
            
        #save the tables:
        np.savez(utils.data_dir + 'optics_tables/' + wl_region + '/' + suffix,
                 optics=np.array(optics_table,dtype='float32').transpose((1,2,0,3,4,5)), 
                 wavelengths=wavelengths, params=['abs','ext','asm'],
                 mode=np.arange(4), ref_index_real=np.array(refrs),
                 ref_index_imag=np.array(refis), surf_mode_radius = rs)


#generate optics tables at many resolutions:
if __name__ ==  '__main__':
    modal_optics_table(nrefr=9  ,nrefi=17 ,nrad=257 ,ndist=33)
    modal_optics_table(nrefr=33 ,nrefi=33 ,nrad=513 ,ndist=65)
    modal_optics_table(nrefr=65 ,nrefi=65 ,nrad=1025,ndist=129)
    modal_optics_table(nrefr=129,nrefi=129,nrad=2049,ndist=257)
    modal_optics_table(nrefr=257,nrefi=257,nrad=2049,ndist=513,test_dataset=True)
