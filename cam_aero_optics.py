#!/usr/bin/env python3
#
#Andrew V. Geiss, Feb 12th 2022
#
#This is a python implementation of the CAM5 aerosol optics parameterization
#described in Ghan and Zaveri, JGR, 2007.

import numpy as np
import optics_utils as utils
from multiprocess import Pool
from os.path import exists as check_path
from numba import njit

#vectorizes input particle properties and does Mie calculations in parallel:
def parallel_mie(radii, wavelength, refr, refi):
    parpool = Pool(utils.n_workers)
    shape = radii.shape
    inputs = np.stack([dat.flatten() for dat in [radii, wavelength, refr, refi]], axis=1)
    outputs = np.array(parpool.starmap(utils.mie,inputs))
    qabs, qsca, g  = [np.reshape(outputs[:,i],shape) for i in range(3)]
    parpool.close()
    return qabs, qsca, g
    
#subroutine for computing a grid of real and imaginary refractive indices
#over which to do mie calculations:
def ref_ind_range(spect_region,nrefr,nrefi,band):
    #load the optical properties of the aerosols and water:
    crefw = utils.read_water_refindex(spect_region)
    crefa = utils.read_modal_optics(spect_region)
    
    #determine the boundaries for the refractive index ranges
    crefs = np.append(np.concatenate(crefa,axis=0)[:,band],crefw[band])
    ibounds = [np.min(np.imag(crefs)), np.max(np.imag(crefs))]
    rbounds = [np.min(np.real(crefs)), np.max(np.real(crefs))]
    
    #define the grid. real indices are regularly spaced, imaginary indices
    #have exponential-spacing
    refi = ibounds[1]*0.3**(np.arange(nrefi,0,-1)-1)
    refi[0] = 0
    refr = np.linspace(*rbounds,nrefr)
    return refr, refi

#computes the chebyshev coefficients for interpolating with respect to surface mode radius.
#fits the data along the last dimension of y
def chebfit(y,ncoef=5):
    shape = y.shape
    coefs = np.zeros((*shape[:-1],ncoef),dtype='float32')
    spc = (np.arange(shape[-1])+0.5)/shape[-1]
    for j in range(ncoef):
        coefs[...,j] = np.sum(y*np.cos(np.pi*j*spc),axis=-1)
    return coefs*2/shape[-1]

#Pre-computes a table of bulk aerosol optical properties that will be used by
#the parameterization
def gen_optics_table(nrefr=7, nrefi=10, nrad=200, ndist=30):

    #define the surface mode radius for each of the distributions, here they
    #are spaced to match Chebyshev nodes
    xr = np.cos(np.pi*(np.arange(0,ndist)+0.5)/ndist)                               #chebyshev points
    rmmin, rmmax = utils.rm_range
    rs = np.exp((xr+1)*0.5*np.log(rmmin) + (1-xr)*0.5*np.log(rmmax))                #surface mode radius
    
    for wl_region, wavelengths in zip(['sw','lw'],[utils.SW_BANDS,utils.LW_BANDS]):
        wavelengths = np.mean(wavelengths,axis=1)
        optics_table, optics_coefs = [], []
        refrs,refis = [],[]
        for band in range(len(wavelengths)):
            print('processing ' + wl_region + ' ' + str(int(np.round(wavelengths[band]*1E9))) + 'nm band')
            
            #define the refractive index and radius ranges:
            refr, refi = ref_ind_range(wl_region, nrefr, nrefi, band)
            refrs.append(refr);refis.append(refi)
            radii = np.exp(np.linspace(np.log(utils.r_range[0]),np.log(utils.r_range[1]),nrad))
            refr,refi,radii = np.meshgrid(refr,refi,radii,indexing='ij')
            
            #compute the table of Mie optical properties:
            wavelength = wavelengths[band]*np.ones(refr.shape)
            qabs, qsca, g = parallel_mie(radii, wavelength, refr, refi)
            
            #integrate optical properties over a variety of size distributions:
            modal_abs, modal_ext, modal_asm = utils.modal_int_lognorm(qabs,qsca,g,radii,rs,mass_scaling=False)
            optics_table.append([modal_abs, modal_ext, modal_asm])
            
            #compute the chebyshev coefficients:
            modal_abs = chebfit(modal_abs); modal_asm = chebfit(modal_asm)      #fit the absorption and asymmetry parameters
            modal_ext = chebfit(np.log(modal_ext))                              #fit the log of the extinction data
            optics_coefs.append([modal_abs, modal_ext, modal_asm])
            
        #save the tables:
        np.save(utils.data_dir + 'optics_tables/' + wl_region + '/cam.npy',np.array(optics_table).transpose((1,2,0,3,4,5)))
        np.savez(utils.data_dir + 'optics_tables/' + wl_region + '/cheb_coefs_cam.npz',
                 cheb_coefs=np.array(optics_coefs).transpose((1,2,0,3,4,5)), 
                 wavlengths=wavelengths, params=['abs','ext','asm'],
                 mode=np.arange(4), ref_index_real=np.array(refrs),
                 ref_index_imag=np.array(refis), surf_mode_radius = rs)
        
#function to do 2-D interpolation of chebyshev coefficients
@njit
def linterp_weights(x,q):
    for i in range(len(x)):
        if x[i] == q:
            return 0.5,0.5,i,i
        if x[i]>q:
            dx = x[i]-x[i-1]
            return i-1,i,(x[i]-q)/dx,(q-x[i-1])/dx

@njit
def binterp(refr, refi, queryr, queryi, coefs):
    assert queryr <= refr[-1] and queryr >= refr[0]
    assert queryi <= refi[-1] and queryi >= refi[0]
    i0r,i1r,w0r,w1r = linterp_weights(refr,queryr)
    i0i,i1i,w0i,w1i = linterp_weights(refi,queryi)
    i0r = int(i0r); i1r = int(i1r); i0i = int(i0i); i1i = int(i1i)
    c = ((coefs[i0r,i0i,:]*w0r + 
          coefs[i1r,i0i,:]*w1r)*w0i + 
          (coefs[i0r,i1i,:]*w0r + 
          coefs[i1r,i1i,:]*w1r)*w1i)
    return c

#performs chebyshev interpolation given coefficients in c for the point x in [-1,1]
@njit
def cheb_interp(c,x):
    assert x>=-1 and x<=1
    #evaluate the chebyshev polynomials (T):
    T = [1, x]
    for i in range(2,len(c)):
        T.append(2*x*T[-1] - T[-2])
    #interpolation step:
    return np.sum(np.array(T)*c)-0.5*c[0]

#this code is meant to be run on import, it either reads the optics table from
#disk or generates it if it is not there:
if not check_path(utils.data_dir + 'optics_tables/sw/cheb_coefs_cam.npz'):
    gen_optics_table()
sw_table = np.load(utils.data_dir + 'optics_tables/sw/cheb_coefs_cam.npz')
lw_table = np.load(utils.data_dir + 'optics_tables/lw/cheb_coefs_cam.npz')

#assign the data into dictionaries:
chebyshev_coefficients = {'sw': np.float64(sw_table['cheb_coefs']), 'lw': np.float64(lw_table['cheb_coefs'])}
refractive_indices = {'sw': {'real': sw_table['ref_index_real'], 'imag': sw_table['ref_index_imag']},
                      'lw': {'real': lw_table['ref_index_real'], 'imag': lw_table['ref_index_imag']}}

def modal_optics(refr,refi,rs,mode,band,wl_region,mass_scaling=True,nrad=200):
    #performs estimate of modal aerosol optical properties.
    #
    #INPUTS:
    #   refr            real refractive index
    #   refi            imaginary refractive index
    #   rs              surface mode radius (m)
    #   mode            mode number (1-4)
    #   band            band number (1-14 sw) and (1-16 lw)
    #   wl_region       'lw' or 'sw'
    #
    #OUTPUTS:
    #   qabs (), qext(), asym (0-1)
    
    assert wl_region == 'sw' or wl_region == 'lw'
    
    #grab the relevant coefficients:
    coefs = chebyshev_coefficients[wl_region][:,mode,band,...]
    refrs = refractive_indices[wl_region]['real'][band,:]
    refis = refractive_indices[wl_region]['imag'][band,:]
    
    #2-D interpolation of the chebyshev coefficients with respect to refractive indices
    #for each of the different optical parameters:
    coefs = [binterp(refrs, refis, refr, refi, coefs[i,:,:,:]) for i in range(3)]
    
    #convert the surface mode radius to the [-1,1] range used by the chebyshev interpolation
    lrmrng = np.log(utils.rm_range)
    rsc = -(2*np.log(rs)-lrmrng[0]-lrmrng[1])/(lrmrng[1]-lrmrng[0])
    
    #do chebyshev interpolation with respect to surface mode radius:
    absp, extp, asym = [cheb_interp(c,rsc) for c in coefs]
    extp = np.exp(extp)
    
    #multiply by the integrated particle mass to get bulk abs and ext efficiencies:
    if mass_scaling:
        rmin, rmax = utils.r_range
        dlogr = np.log(rmax/rmin)/(nrad-1)
        radii = np.exp(np.linspace(np.log(rmin),np.log(rmax),nrad))
        ds = np.exp(-0.5*(np.log(radii/rs)/utils.alnsg_amode[mode])**2)*dlogr
        masswet = utils.rho_h2o*np.sum(radii*ds,axis=-1)*4/3
        absp *= masswet
        extp *= masswet
    
    return absp, extp, asym
