#!/usr/bin/env python3
#
#plot_jhist.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cheb = np.concatenate((np.load('../data/predictions/test/sw_cheb.npy'),
                       np.load('../data/predictions/test/lw_cheb.npy')[:,np.newaxis]),axis=-1)
ann = np.concatenate((np.load('../data/predictions/test/sw_ann.npy'),
                       np.load('../data/predictions/test/lw_ann.npy')[:,np.newaxis]),axis=-1)
targ = np.concatenate((np.load('../data/testing_data/sw.npy')[:,-3:],
                       np.load('../data/testing_data/lw.npy')[:,-3][:,np.newaxis]),axis=-1)
nsamples = targ.shape[0]

def bin_centers(x):
    return (x[1:] + x[:-1])/2

var_names = ['SW Absorption', 'SW Extinction', 'SW Asymmetry', 'LW Absorption']
fig = plt.figure(figsize=(17,4))
for var in range(4):
    plt.subplot(1,4,var+1)
    xmax = np.max(targ[:,var])
    counts, x, _ = np.histogram2d(targ[:,var],cheb[:,var],bins=200,range=[[0,xmax],[0,xmax]])
    pc = plt.pcolormesh(x,x,100*counts,norm=colors.LogNorm(1,100000),cmap='Greys')
    counts = np.histogram2d(ann[:,var],targ[:,var],bins=200,range=[[0,xmax],[0,xmax]])[0]
    x = np.array([0] + list(bin_centers(x)) + [xmax])
    counts = np.pad(counts,[[1,1],[1,1]],'constant',constant_values=0)
    plt.contour(x,x,counts,levels = [np.min(counts.flatten()[counts.flatten()>0])-1E-16],colors='r',linewidths=[0.8])
    if var == 0:
        plt.ylabel('Parameterization')
    plt.xlabel('Mie Code')
    plt.title(var_names[var])
    plt.xlim(0,xmax)
    plt.ylim(0,xmax)
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.2, 0.01, 0.6])
cb = fig.colorbar(pc,cax=cbar_ax)
cb.ax.set_title('Counts',va='bottom')

plt.savefig('../figures/error_jhists.png',dpi=500,bbox_inches='tight')