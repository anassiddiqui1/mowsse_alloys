import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssl
import os
import warnings
warnings.filterwarnings('ignore')

fontsize = 8.3
plt.rcParams['font.size'] = fontsize
plt.rcParams["figure.figsize"] = (4.75,3.75)
from matplotlib.ticker import MultipleLocator

def gaussian(x,amplitude,mean,std):
    return amplitude*np.exp(-(x-mean)**2/(2*std**2))

def get_spectra_curve(x,intensities,frequencies,std):
    
    y = np.zeros(len(x))

    for i,intensity in enumerate(intensities):
        y = y + gaussian(x,intensity,frequencies[i],std)
    return x,y

def get_max_peak(xdata,ydata,xmin,xmax):
    
    idx = np.where((xdata>xmin) & (xdata<xmax))[0]
    peaks = ssl.find_peaks(ydata[idx])[0]
    #print(peaks[0])
    biggest_peak_idx = peaks[np.argmax(ydata[idx][peaks])]
    biggest_peak_xdata = xdata[idx][biggest_peak_idx]
    biggest_peak_ydata = ydata[idx][biggest_peak_idx]
    return biggest_peak_xdata,biggest_peak_ydata

fig,axs = plt.subplots(3,1,figsize=(4.75,7.5),gridspec_kw={'height_ratios': [2,1,1]})

#ncells = [4,5,6,7,8,9,10,12,15]
ncells = [4,6,8,10,15]
colors = ['blue','green','red','orange','yellow']
# alphas=[1,0.9,0.7,0.2,0.3]
std = 3
xspace = np.linspace(0,500,5000)
npeaks = 3
nconfig=15

peaks = np.zeros((npeaks,len(ncells),2))

for i,ncell in enumerate(ncells):
    avg_intens = np.zeros(len(xspace))
    for config in range(nconfig):

        filename = f'cellsize_convergence_data/ncell_{ncell}_config_{config}.npz'
        npzfile = np.load(filename)
        intens,freqs = npzfile['intensities'],npzfile['freqs']
        avg_intens = avg_intens + get_spectra_curve(xspace,intens,freqs,std)[1]
    avg_intens /= nconfig
    axs[0].plot(xspace,avg_intens,label=f'$N$={ncell}x{ncell}',color=colors[i])

    peaks[0,i,:] = get_max_peak(xspace,avg_intens,200,300)
    peaks[1,i,:] = get_max_peak(xspace,avg_intens,300,370)
    peaks[2,i,:] = get_max_peak(xspace,avg_intens,370,420)

axs[0].arrow(peaks[0,-1,0]-30,peaks[0,-1,1]+10,28,-8,width=0.5)
axs[0].text(peaks[0,-1,0]-90,peaks[0,-1,1]+11,'Peak 1')

axs[0].arrow(peaks[1,-1,0]-30,peaks[1,-1,1]+10,25,-5,width=0.5)
axs[0].text(peaks[1,-1,0]-90,peaks[1,-1,1]+11,'Peak 2')

axs[0].arrow(peaks[2,-1,0]-30,peaks[2,-1,1]+5,23,-3,width=0.5)
axs[0].text(peaks[2,-1,0]-90,peaks[2,-1,1]+5,'Peak 3')

ml = MultipleLocator(10)
axs[0].xaxis.set_minor_locator(ml)
axs[0].legend(frameon=False,loc='upper left')
axs[0].set_xlabel(r'Raman shift (cm$^{-1}$)')
axs[0].set_ylabel('Intensities (a.u.)')
axs[0].text(-0.08,0.98,'(a)',fontsize=fontsize+4,fontweight='extra bold',transform=axs[0].transAxes)

axs[1].plot(ncells,peaks[0,:,0],marker='o',c='b')
axs[1].axhline(peaks[0,-1,0],c='k',linestyle='--')
axs[1].text(min(ncells),peaks[0,-1,0]+9,'Peak 1',c='b')
axs[1].plot(ncells,peaks[1,:,0],marker='o',c='g')
axs[1].axhline(peaks[1,-1,0],c='k',linestyle='--')
axs[1].text(min(ncells),peaks[1,-1,0]-20,'Peak 2',c='g')
axs[1].plot(ncells,peaks[2,:,0],marker='o',c='r')
axs[1].axhline(peaks[2,-1,0],c='k',linestyle='--')
axs[1].text(min(ncells),peaks[2,-1,0]-20,'Peak 3',c='r')
axs[1].set_ylabel(r'Raman shift (cm$^{-1}$)')
axs[1].set_xlabel('$N$')
ml = MultipleLocator(10)
axs[1].yaxis.set_minor_locator(ml)
axs[1].text(-0.08,0.95,'(b)',fontsize=fontsize+4,fontweight='extra bold',transform=axs[1].transAxes)

axs[2].plot(ncells,peaks[0,:,1],marker='o',c='b')
axs[2].axhline(peaks[0,-1,1],c='k',linestyle='--')
axs[2].text(min(ncells),peaks[0,-1,1]+2,'Peak 1',c='b')
axs[2].plot(ncells,peaks[1,:,1],marker='o',c='g')
axs[2].axhline(peaks[1,-1,1],c='k',linestyle='--')
axs[2].text(min(ncells),peaks[1,-1,1]-6,'Peak 2',c='g')
axs[2].plot(ncells,peaks[2,:,1],marker='o',c='r')
axs[2].axhline(peaks[2,-1,1],c='k',linestyle='--')
axs[2].text(min(ncells),peaks[2,-1,1]-6,'Peak 3',c='r')
axs[2].set_ylabel('Intensity (a.u.)')
axs[2].set_xlabel('$N$')
axs[2].yaxis.set_minor_locator(MultipleLocator(5))
axs[2].text(-0.08,0.99,'(c)',fontsize=fontsize+4,fontweight='extra bold',transform=axs[2].transAxes)

plt.tight_layout()
plt.savefig('freqs_intens_vs_ncells',dpi=300,facecolor='white',transparent=False,
           bbox_inches='tight',pad_inches=0.02)
