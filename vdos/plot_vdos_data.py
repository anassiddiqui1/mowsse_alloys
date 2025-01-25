import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import sys
from pathlib import Path
repo_path = Path("../utils")
sys.path.append(str(repo_path))
from tools import get_color

fontsize = 8.3
#fontsize = 4
plt.rcParams['font.size'] = fontsize
plt.rcParams["figure.figsize"] = (4.75,3.75)

#Range of x and y for alloy compositions
xs = np.linspace(0,1,5)
ys = np.linspace(0,1,5)

fig,axs = plt.subplots(5,5)

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        color = get_color(x,y)
        dos = np.loadtxt(f'vdos_data/x{x:5.3f}y{y:5.3f}dossum.dat')
        
        if (x,y) in [(0,0),(0,1),(1,0),(1,1)]:
            nconfigs = 1
        else:
            nconfigs = 15
        ###Plot average density of states
        axs[i][j].plot(dos/nconfigs,color=color)
        
        axs[i][j].set_ylim(-1,56)
        axs[i][j].xaxis.set_minor_locator(MultipleLocator(50))
        axs[i][j].xaxis.set_major_locator(MultipleLocator(200))
        axs[i][j].set_yticks([])
        if j==0:
            axs[i][j].set_ylabel(f'x={x}')
        
        if i!=4:
            axs[i][j].tick_params(labelbottom=False)
        
        if i==0:
            axs[i][j].set_xlabel(f'y={y}')
            axs[i][j].xaxis.set_label_position('top') 
            
fig.text(0.5, 0.04, r"$\mathrm{\omega}$ (cm$^{-1}$)", ha='center')
fig.text(0.06, 0.5, 'Density of States (a.u.)', va='center', rotation='vertical')
plt.subplots_adjust(wspace=0.05, hspace=0.15)
plt.savefig('vibrational_dos_mace_demo.png',dpi=300,facecolor='white',transparent=False,
            bbox_inches='tight',pad_inches=0.02)
plt.show()
