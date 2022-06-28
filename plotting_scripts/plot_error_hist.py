#!/usr/bin/env python3
#
#plot_error_hist.py

import numpy as np
import matplotlib.pyplot as plt

#load in the test set data and predictions:
pred_types = ['129,129,2049,257','9,17,257,33','33,33,513,65','65,65,1025,129', 'ann','cheb']
llabels = [None, None, None,'Tables', 'ANN', 'EAM']
sw_preds = np.array([np.load('../data/predictions/test/sw_' + p + '.npy') for p in pred_types])
lw_preds = np.array([np.load('../data/predictions/test/lw_' + p + '.npy') for p in pred_types])
preds = np.concatenate((sw_preds,lw_preds[...,np.newaxis]),axis=2)
targs = np.concatenate((np.load('../data/random_test_set_sw.npy')[:,-3:],
                        np.load('../data/random_test_set_lw.npy')[:,-3][:,np.newaxis]),axis=-1)

titles = ['SW Absorption','SW Extinction','SW Asymmetry','LW Absorption']
c = 0.5
line_colors = [[c,c,c], [c,c,c],[c,c,c],[c,c,c], '#BB5566','#004488']
plt.figure(figsize=(16,4))
for var in range(4):
    plt.subplot(1,4,var+1)
    xmin = 100
    xmax = 0
    for method in range(6):
        mae = np.abs(preds[method,:,var] - targs[:,var])
        counts,bins = np.histogram(mae,100)
        bins = (bins[1:] + bins[:-1])/2
        if method>3:
            plt.loglog(bins,counts,color=line_colors[method],linewidth=2,label = llabels[method])
        else:
            plt.loglog(bins,counts,'--',color=line_colors[method],linewidth=2,label = llabels[method])
        xmin = min(xmin,bins[0])
        xmax = max(xmax,bins[-1])
    plt.xlim(xmin,xmax)
    plt.ylim(10,10**6)
    plt.grid(True,which='both',color=[0.9,0.9,.9])
    plt.xlabel('MAE')
    if var == 0:
        plt.ylabel('Counts')
        plt.legend()
    plt.title(titles[var])
plt.savefig('../figures/error_hists.png',dpi=500,bbox_inches='tight')