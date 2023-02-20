#!/usr/bin/env python3
#
#plot_error_hist.py

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

#set some constants:
pred_types = ['129,129,2049,257','9,17,257,33','33,33,513,65','65,65,1025,129', 'ann','cheb']
llabels = [None, None, None,'Tables', 'ANN', 'EAM']
titles = ['SW Absorption','SW Extinction','SW Asymmetry','SW Scattering','SSA','LW Absorption']
c = 0.5
line_colors = [[c,c,c], [c,c,c],[c,c,c],[c,c,c], '#BB5566','#004488']
bins = 10**(np.arange(-10,2,.5))
centers = (bins[1:] + bins[:-1])/2
bins[0] = 0
bins[-1] = 1000

#function to compute histograms (numpy is slow)
@njit
def hist(x,bins):
    N = len(x)
    NB = len(bins)
    counts = np.zeros((NB-1,),dtype='double')
    for i in prange(N):
        dx = x[i]
        if np.isnan(dx):
            continue
        ix = 1
        while bins[ix] < dx:
            ix += 1
        counts[ix] += 1
    return counts

#function to pre-compute the error histograms:
def calc_histograms():
    histograms = np.zeros((len(pred_types),len(titles),len(centers)))
    truth = np.load('../data/evaluation/test_sw_mie.npy')
    for ip in range(len(pred_types)):
        pred = np.load('../data/evaluation/test_sw_' + pred_types[ip] + '.npy')
        for iv in range(len(titles)-1):
            abserr = np.abs(truth[:,iv]-pred[:,iv])
            histograms[ip,iv,:] = hist(abserr,bins)
    
    truth = np.load('../data/evaluation/test_lw_mie.npy')
    for ip in range(len(pred_types)):
        pred = np.load('../data/evaluation/test_lw_' + pred_types[ip] + '.npy')
        abserr = np.abs(truth-pred)
        histograms[ip,-1,:] = hist(abserr,bins)
    np.save('pre_calced_histograms.npy',histograms)
    
# calc_histograms()
histograms = np.load('pre_calced_histograms.npy')

plt.figure(figsize=(16,11))

ymin = 10
ymax = 100_000_000
grid_color = .8*np.array([1,1,1])
for var in range(6):
    plt.subplot(2,3,var+1)
    xmin = 100
    xmax = 0
    for method in range(6):
        for i in range(len(bins)):
            plt.plot([bins[i],bins[i]],[ymin,ymax],'-',linewidth=0.5,color=grid_color)
        counts = histograms[method,var,:]
        if method>3:
            plt.loglog(centers,counts,color=line_colors[method],linewidth=2,label = llabels[method])
        else:
            plt.loglog(centers,counts,'--',color=line_colors[method],linewidth=2,label = llabels[method])
    plt.xlim(10**-7,10)
    plt.ylim(ymin,ymax)
    if var>2:
        plt.xlabel('MAE')
    if var == 0 or var == 3:
        plt.ylabel('Counts')
    if var == 0:
        plt.legend()
    plt.title(titles[var])
plt.savefig('../figures/error_hists.png',dpi=500,bbox_inches='tight')