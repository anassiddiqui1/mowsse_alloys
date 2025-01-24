import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression

fontsize =8.3
plt.rcParams['font.size'] = fontsize
fig,axs = plt.subplots(2,1,figsize=(4.75,7.5))

#Read W-W data from npz file
data = np.load('binding_energy_data/MoWS2_nn.npz')
n_W_pairs = data['nns']
energies = data['energies']

#Calculate E_W-W using mean of energy/n_W-W
Eb = np.mean(energies/n_W_pairs)

axs[0].scatter(n_W_pairs,(energies-n_W_pairs*Eb)/n_W_pairs)
axs[0].set_xlabel('Number of W-W NN pairs')
axs[0].set_ylabel(r'E$_{\rm b}$ difference/pair (eV)')
axs[0].set_xlim(8,25)
axs[0].xaxis.set_minor_locator(MultipleLocator(1))
axs[0].text(-0.2,0.98,'(a)',fontsize=fontsize+4,fontweight='extra bold',transform=axs[0].transAxes)
#Read W-W data from npz file
data = np.load('binding_energy_data/MoSSe_nn.npz')
n_Se_pairs = data['nns']
n_Se_z_pairs = data['nns_vertical']
n_Se_plane_pairs = data['nns_lateral']
energies = data['energies']

#Use 2D linear regression to get Eb_z and Eb_plane separately
regr = LinearRegression(fit_intercept=False)
x = np.column_stack((n_Se_z_pairs,n_Se_plane_pairs))
y = energies
regr.fit(x, y)
Eb_z,Eb_plane = regr.coef_

axs[1].scatter(n_Se_pairs,(energies-n_Se_plane_pairs*Eb_plane-n_Se_z_pairs*Eb_z)/n_Se_pairs)
axs[1].set_xlabel('Number of Se-Se NN pairs')
axs[1].set_ylabel(r'E$_{\rm b}$ difference/pair (eV)')
axs[1].xaxis.set_minor_locator(MultipleLocator(1))
axs[1].text(-0.2,0.98,'(b)',fontsize=fontsize+4,fontweight='extra bold',transform=axs[1].transAxes)

plt.tight_layout()

plt.savefig('binding_energy_plots.png',facecolor='white', 
            transparent=False,dpi=300,bbox_inches='tight',pad_inches=0.02)

plt.show()
