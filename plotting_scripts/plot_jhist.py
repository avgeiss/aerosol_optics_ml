import numpy as np
from numba import njit, prange
from os.path import exists

#compute or load the pre-computed joint-histograms
@njit(parallel=True)
def hist2d(x,y,bins):
    N = len(x)
    NB = len(bins)
    counts = np.zeros((NB-1,NB-1),dtype='double')
    for i in prange(N):
        dx, dy = x[i], y[i]
        if np.isnan(dx) or np.isnan(dy):
            continue
        ix, iy = 1, 1
        while bins[ix] < dx:
            ix += 1
        while bins[iy] < dy:
            iy += 1
        counts[ix-1,iy-1] += 1
    return counts

if not exists('../data/evaluation/error_joint_histograms.npz'):    
    histograms_cheb, histograms_ann, bins = [], [], []
    pred_cheb = np.load('../data/evaluation/test_sw_cheb.npy')
    pred_ann = np.load('../data/evaluation/test_sw_ann.npy')
    truth = np.load('../data/evaluation/test_sw_mie.npy')
    for i in range(5):
        bins.append(np.linspace(0,np.nanmax(truth[:,i]),129))
        histograms_cheb.append(hist2d(pred_cheb[:,i],truth[:,i],bins[-1]))
        histograms_ann.append(hist2d(pred_ann[:,i],truth[:,i],bins[-1]))
    pred_cheb = np.load('../data/evaluation/test_lw_cheb.npy')
    pred_ann = np.load('../data/evaluation/test_lw_ann.npy')
    truth = np.load('../data/evaluation/test_lw_mie.npy')  
    bins.append(np.linspace(0,np.nanmax(truth),129))
    histograms_cheb.append(hist2d(pred_cheb, truth, bins[-1]))
    histograms_ann.append(hist2d(pred_ann, truth, bins[-1]))
    np.savez('../data/evaluation/error_joint_histograms.npz',
             histograms_cheb = np.array(histograms_cheb),
             histograms_ann = np.array(histograms_ann),
             bins = np.array(bins))
dataset = np.load('../data/evaluation/error_joint_histograms.npz')
histograms_cheb = dataset['histograms_cheb']
histograms_ann = dataset['histograms_ann']
bins = dataset['bins']
centers = (bins[:,1:] + bins[:,:-1])/2

#do plotting:
import matplotlib.pyplot as plt
from matplotlib import colors
var_names = ['SW Absorption', 'SW Extinction', 'SW Asymmetry', 'SW Scattering', 'SW SSA', 'LW Absorption']
fig = plt.figure(figsize=(15,10))
for var in range(6):
    cheb, ann = histograms_cheb[var,:,:], histograms_ann[var,:,:]
    plt.subplot(2,3,var+1)
    pc = plt.pcolormesh(centers[var,:],centers[var,:],cheb,norm=colors.LogNorm(10,1E6),cmap='Greys',shading='nearest')
    x = np.concatenate([[bins[var,0]],centers[var,:],[bins[var,-1]]])
    ann = np.pad(ann,[[1,1],[1,1]],'constant',constant_values=0)
    plt.contour(x,x,ann,levels = [np.min(ann.flatten()[ann.flatten()>0])-1E-16],colors='r',linewidths=[0.8])
    if var == 0 or var == 3:
        plt.ylabel('Parameterization')
    if var>2:
        plt.xlabel('Mie Code')
    plt.title(var_names[var])
    plt.xlim(0,centers[var,-1])
    plt.ylim(0,centers[var,-1])
    if var == 0 or var == 5:
        plt.xticks([0,.2,.4,.6,.8,1.0,1.2,1.4])
        plt.yticks([0,.2,.4,.6,.8,1.0,1.2,1.4])
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.2, 0.01, 0.6])
cb = fig.colorbar(pc,cax=cbar_ax,extend='both')
cb.ax.set_title('Counts',va='bottom')
plt.savefig('../figures/error_jhists.png',dpi=500,bbox_inches='tight')


