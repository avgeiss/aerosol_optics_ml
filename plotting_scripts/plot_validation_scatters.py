import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
targ_param_counts = np.array([500,1000,2500,5000,7500,10_000,15_000,25_000,50_000,100_000])
colors = ['#999999','#CC6677', '#44AA99', '#117733', '#AA4499', '#999933']

def plot_errors(ax,wvl_region,zoom=False):

    #load and plot the scatter showing the random ANN performance:
    data = np.load('../data/predictions/validation/' + wvl_region + '_validation.npz')
    params = data['params']
    error = data['error']
    if zoom:
        ax.plot(params,error,'k.',markersize=6)
    else:
        ax.plot(params,error,'k.',markersize=2)
    
    
    bench_loss = np.load('../data/predictions/validation/bench_valid_errors_' + wvl_region + '.npy')
    sizes = np.array([500,1000,2500,5000,7500,10_000,15_000,25_000,50_000,100_000])
    loss = np.nanmin(bench_loss,axis=-1)
    for i in range(1,6):
        if zoom:
            ax.plot(sizes,loss[i,:],'-',color=colors[i],linewidth=3,alpha=0.8)
        else:
            ax.plot(sizes,loss[i,:],'-',color=colors[i],linewidth=1.5,alpha=0.8)
    
    return None

plt.figure(figsize=(15,4.5))
wvl_regions = ['sw','lw1','lw2']
titles = ['Shortwave','Longwave-1','Longwave-2']
sp_count = 1
xlims = [[0,100_000],[0,100_000],[0,100_000]]
ylims = [[0,9E-4],[0,4.5E-4],[0,4.5E-4]]
inxl = [[0,20_000],[0,20_000],[0,20_000]]
inyl = [[1.25E-4,2E-4],[3.5E-5,1E-4],[3.5E-5,1E-4]]
for wvl_region in wvl_regions:
    ax = plt.subplot(1,3,sp_count)
    plot_errors(ax, wvl_region)
    xl = inxl[sp_count-1]
    yl = inyl[sp_count-1]
    plt.xlim(xlims[sp_count-1])
    plt.ylim(ylims[sp_count-1])
    plt.xlabel('Model Parameters')
    plt.ylabel('Validation MAE (dimensionless)')
    plt.title(titles[sp_count-1])
    plt.xticks(ticks=[0,20_000,40_000,60_000,80_000,100_000],labels=['0','20k','40k','60k','80k','100k'])
    plt.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
    if wvl_region=='sw':
        leg = plt.legend(['Random','2-Layers','3-Layers','4-Layers','5-Layers','6-Layers'],loc='lower center',ncol=3,borderpad=0.2,borderaxespad=0.2,fontsize=9)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
    rect = Rectangle((xl[0],yl[0]),xl[1]-xl[0],yl[1]-yl[0], linewidth=1, edgecolor='k', facecolor=[.8,.8,.8,.8])
    ax.add_patch(rect)
    if wvl_region == 'sw':
        plt.annotate(text='', xy=(33_000,2.98E-4), xytext=(21_000,2.03E-4), arrowprops=dict(arrowstyle='->',linewidth=2))
    else:
        plt.annotate(text='', xy=(33_000,1.5E-4), xytext=(21_000,1.02E-4), arrowprops=dict(arrowstyle='->',linewidth=2))
    
        
        
    inax = inset_axes(ax, width='60%', height='60%', loc='upper right', borderpad=1.5)
    inax.set_facecolor((.8,.8,.8))
    if wvl_region == 'sw':
        inax.plot(7895,0.00013596652,'ro')
    elif wvl_region == 'lw1':
        inax.plot(4858,4.666972E-05,'ro')
    elif wvl_region == 'lw2':
        inax.plot(5336,4.5627672E-05,'ro')
    plot_errors(inax, wvl_region, zoom=True)
    plt.xlim(inxl[sp_count-1])
    plt.ylim(inyl[sp_count-1])
    plt.xticks(ticks=[0,5_000,10_000,15_000,20_000],labels=['0','5k','10k','15k','20k'],fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('',fontsize=8)
    inax.yaxis.offsetText.set_fontsize(8)
    plt.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
    sp_count += 1

plt.savefig('../figures/random_ann_scatter.png',dpi=500,bbox_inches='tight')