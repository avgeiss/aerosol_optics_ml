#!/usr/bin/env python3
#
#Andrew V. Geiss, Feb 8th 2022
#
# This contains subroutines and important constants (such as CAM5 spectral bands, particle radius
# ranges, modal size distribution log-standard deviations, and water density) that get used by
# both 'cam_aero_optics.py' and 'create_optics_tables.py.' The subroutines are for reading rrtmg optics
# files, integrating optical properties over particle size distributions, and a wrapper for MieQ
#

import numpy as np
from PyMieScatt import MieQ
from netCDF4 import Dataset

#define constants:
n_workers = 24                                                  #my system has 12-cores with 2-threads ea. change as needed
data_dir = './data/'                                            #data location
SIGMAG_AMODE = np.double([1.800, 1.600, 1.800, 1.600])          #geometric standard deviation of MAM aerosol modes
rm_range =  [0.01E-6, 25.0E-6]                                  #range of size distributions
r_range =   [0.001E-6, 100.0E-6]                                #range of particle radii
alnsg_amode = np.log(np.double([1.800, 1.600, 1.800, 1.600]))   #log standard deviation assumed for each mode:
rho_h2o = 1E3                                                   #density of water kg/m^3

#The optical bands used in E3SM by wavelength in meters:
SW_BANDS = 0.01/np.array([[2600,3250,4000,4650,5150,6150,7700,8050,12850,16000,22650,29000,38000,820],
            [3250,4000,4650,5150,6150,7700,8050,12850,16000,22650,29000,38000,50000,2600]],dtype='double').T
LW_BANDS = 0.01/np.array([[10,350,500,630,700,820,980,1080,1180,1390,1480,1800,2080,2250,2390,2600],
            [350,500,630,700,820,980,1080,1180,1390,1480,1800,2080,2250,2390,2600,3250]],dtype='double').T

#Performs Mie scattering calculations. This is a wrapper for the MieQ function
#from PyMieScatt. I have validated the output from this function against 
#Wiscombe's Fortran Mie code for the inputs used in Ghan et al.'s original
#parameterization with the following error (that should be negligible compared 
#to other sources of error in the parameterization):
#
#                        QABS         QSCA        ASYMM
#   Max Abs. Err.  0.00184618   0.01585432   0.00964629  
#   99.9 Prctl.    0.00114451   0.00149886   0.00079685
#   99 Prctl.      0.00032988   0.00043783   0.00044239
#
def mie(radius, wavelength, refr, refi):                                        #radius and wavelength should be expressed in meters
    size_param = min(2*np.pi*radius/wavelength,400.0)                           #this appears to be a kluge I've carried over from ghan and zaveri's code
    radius = size_param*wavelength/(2*np.pi)                                    #that caps the size parameter at 400. can probably be removed
    mie_out = MieQ(refr + refi*1j, wavelength*1E9, 2*radius*1E9)                #PyMieScatt expects wavelength and diameter in nm
    qsca, qabs, g = mie_out[1:4]
    return qabs, qsca, g

#averages aerosol optical properties over a set of lognormal size
#distributions
def modal_int_lognorm(qabs, qsca, g, radii, rs, mass_scaling=False):
    ndist = len(rs)
    
    #preallocate outputs
    modal_abs = np.zeros((len(alnsg_amode),*qabs.shape[:-1],ndist),dtype='float32')
    modal_ext, modal_asm = np.copy(modal_abs), np.copy(modal_abs)
    
    #spacing of log-radii:
    dlogr = np.log(r_range[1]/r_range[0])/(radii.shape[-1]-1)
    
    for m in range(len(alnsg_amode)):       #iterate over the aerosol modes
        for n in range(ndist):              #iterate over the size distributions
            C = 1/(alnsg_amode[m]*np.sqrt(np.pi*2))
            ds = np.exp(-0.5*(np.log(radii/rs[n])/alnsg_amode[m])**2)*dlogr
            masswet = C*rho_h2o*np.sum(radii[0,0,:]*ds[0,0,:])*4/3
            sumabs = C*np.sum(qabs*ds,axis=-1)
            sumsca = C*np.sum(qsca*ds,axis=-1)
            modal_asm[m,...,n] = C*np.sum(g*qsca*ds,axis=-1)/sumsca
            modal_abs[m,...,n] = sumabs
            modal_ext[m,...,n] = sumsca+sumabs
            if mass_scaling:
                modal_abs[m,...,n] /= masswet
                modal_ext[m,...,n] /= masswet
    
    return modal_abs, modal_ext, modal_asm
    
#loads the refractive indices of water used by RRTMG:
def read_water_refindex(spect_region):
    assert (spect_region=='sw') or (spect_region == 'lw')
    ncf = Dataset(data_dir + 'physprops/water_refindex_rrtmg_c080910.nc')
    ref_ind = ncf.variables['refindex_real_water_' + spect_region][:].data + np.abs(ncf.variables['refindex_im_water_' + spect_region][:].data)*1j
    ncf.close()
    return ref_ind

#read in the refractive indices of the aerosol species used by RRTMG:
def read_modal_optics(spect_region):
    assert (spect_region=='sw') or (spect_region == 'lw')
    fnames = [['sulfate_rrtmg_c080918','ocpho_rrtmg_c130709','ocphi_rrtmg_c100508','bcpho_rrtmg_c100508','dust_aeronet_rrtmg_c141106','ssam_rrtmg_c100508','poly_rrtmg_c130816'],
             ['sulfate_rrtmg_c080918','ocphi_rrtmg_c100508','ssam_rrtmg_c100508','poly_rrtmg_c130816'],
             ['ssam_rrtmg_c100508','sulfate_rrtmg_c080918','bcpho_rrtmg_c100508','ocpho_rrtmg_c130709','ocphi_rrtmg_c100508','poly_rrtmg_c130816'],
             ['ocpho_rrtmg_c130709','bcpho_rrtmg_c100508','poly_rrtmg_c130816']]
    cref_inds = []
    for mode_files in fnames:
        modal_cref = []
        for spec_file in mode_files:
            ncf = Dataset(data_dir +  'physprops/' + spec_file + '.nc')
            modal_cref.append(ncf.variables['refindex_real_aer_' + spect_region][:].data + 
                              np.abs(ncf.variables['refindex_im_aer_' + spect_region][:].data)*1j)
            ncf.close()
        cref_inds.append(np.array(modal_cref))
    return cref_inds
