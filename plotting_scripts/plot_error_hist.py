#!/usr/bin/env python3
#
#plot_error_hist.py

import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from numba import njit
from multiprocess import Pool
pred_types = ['9,17,257,33','33,33,513,65','65,65,1025,129', '129,129,2049,257', 'ann','cheb']
titles = ['SW Absorption','SW Extinction','SW Asymmetry','SW Scattering','SSA','LW Absorption']

#a custom histogram function that is much faster than numpy:
@njit
def fast_histogram(data,bins):
    counts = np.zeros((len(bins)-1,),dtype='double')
    N = len(data)
    for i_d in range(N):
        d = data[i_d]
        if np.isnan(d):
            continue
        i_b = 1
        while bins[i_b] < d:
            i_b += 1
        counts[i_b-1] += 1
    return counts

#define the histogram
bins = 10**(np.arange(-10,2,.5))
centers = (bins[1:] + bins[:-1])/2
bins[0] = 0
bins[-1] = 1000
def histogram(data):
    return fast_histogram(data,bins)

def histogram_slow(data):
    hist, _ = np.histogram(data,bins)
    return hist

def compute_error_histograms(pred_type):
    print('Computing errors for ' + pred_type)
    truth_sw = np.load('../data/evaluation/test_sw_mie.npy')
    pred_sw = np.load('../data/evaluation/test_sw_' + pred_type + '.npy')
    errors = list(np.abs(pred_sw-truth_sw).T)
    truth_lw = np.load('../data/evaluation/test_lw_mie.npy')
    pred_lw = np.load('../data/evaluation/test_lw_' + pred_type + '.npy')
    errors.append(np.abs(truth_lw-pred_lw))
    p = Pool(len(errors))
    hists = np.array(p.map(histogram, errors))
    p.close()
    return hists

if exists('../data/evaluation/error_histograms.npy'):
    hists = np.load('../data/evaluation/error_histograms.npy')
else:
    hists = np.array([compute_error_histograms(model_type) for model_type in pred_types])
    np.save('../data/evaluation/error_histograms.npy', hists)
    
#plotting code:
titles = ['SW Absorption','SW Extinction','SW Asymmetry','SW Scattering','SSA','LW Absorption']
c = 0.5
line_colors = [[c,c,c], [c,c,c],[c,c,c],[c,c,c], '#BB5566','#004488']
plt.figure(figsize=(16,11))
bins = 10**(np.arange(-10,2,.5))
centers = (bins[1:] + bins[:-1])/2
bins[0] = 0
bins[-1] = 1000
ymin = 100
ymax = 1_000_000
grid_color = .8*np.array([1,1,1])
for var in range(6):
    plt.subplot(2,3,var+1)
    xmin = 100
    xmax = 0
    for method in range(6):
        for i in range(len(bins)):
            plt.plot([bins[i],bins[i]],[ymin,ymax],'-',linewidth=0.5,color=grid_color)
        mae = np.abs(preds[method,:,var] - targs[:,var])
        counts,_ = np.histogram(mae[np.logical_not(np.isnan(mae))],bins)
        if method>3:
            plt.loglog(centers,counts,color=line_colors[method],linewidth=2,label = llabels[method])
        else:
            plt.loglog(centers,counts,'--',color=line_colors[method],linewidth=2,label = llabels[method])
        xmin = min(xmin,bins[0])
        xmax = max(xmax,bins[-1])
    plt.xlim(10**-9,10)
    plt.ylim(ymin,ymax)
    if var>2:
        plt.xlabel('MAE')
    if var == 0 or var == 3:
        plt.ylabel('Counts')
    if var == 0:
        plt.legend()
    plt.title(titles[var])
plt.savefig('../figures/error_hists_new.png',dpi=500,bbox_inches='tight')
