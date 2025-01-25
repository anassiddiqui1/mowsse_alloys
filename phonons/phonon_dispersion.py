import numpy as np;
import string
from ase.phonons import Phonons;
from ase.optimize import BFGS;
from ase.build.surface import mx2;
import matplotlib.pyplot as plt;
from ase.calculators.espresso import Espresso
import sys
from pathlib import Path
repo_path = Path("../utils")
sys.path.append(str(repo_path))
from tools import mx2, a_xy, t_xy, calc, passlog,relax_structure
import warnings
warnings.filterwarnings('ignore')

lat_param = {'a_MoS2': a_xy(0,0),'t_MoS2': t_xy(0,0),'a_WS2' : a_xy(1,0), 't_WS2' : t_xy(1,0),
            'a_MoSe2': a_xy(0,1),'t_MoSe2': t_xy(0,1),'a_WSe2' : a_xy(1,1), 't_WSe2' : t_xy(1,1)}
calc_espr = None #Change to Espresso calc for rerunning the Espresso phonon calculation

def get_phonon_bandstructure(atoms,N,calc,delta,phonon_name,read=True):

    ph = Phonons(atoms,calc,supercell=(N, N, 1), delta=delta, name=phonon_name); 
    if not read:
        #Relax the structure
        relax_structure(prim_opt,calc,passlog,fmax=0.001)
        ph.clean()
        ph.run()
    ph.read(acoustic=True)
    path = atoms.cell.bandpath('GMKG', npoints=150)
    bs = ph.get_band_structure(path,verbose=False); 
    dos = ph.get_dos(kpts=(60, 60, 1)).sample_grid(npts=1000, width=2e-4)
    return bs,dos

def plot_phonon_bandstructure(ax,dosax,bs,dos,emin,emax,color,label):
    cm1 = 8065.54011 #Conversion factor eV to cm-1
    bs._energies *= cm1
    bs.plot(ax=ax, emin=emin, emax=emax, color=color, ylabel=r"$\mathrm{\omega}$ (cm$^{-1}$)",label=label);
    dosax.plot(dos.get_weights(), dos.get_energies()*cm1,color=color)
    return dos

fig = plt.figure(figsize=(9.5,7))
fontsize = 8.5
plt.rcParams['font.size'] = fontsize
#Dispersion and DOS axes
ax_rects = [[0.07,0.52,0.33,0.43],
           [0.56,0.52,0.33,0.43],
           [0.07,0.04,0.33,0.43],
           [0.56,0.04,0.33,0.43]]

dosax_rects = [[0.41,0.52,0.08,0.43],
           [0.9,0.52,0.08,0.43],
           [0.41,0.04,0.08,0.43],
           [0.9,0.04,0.08,0.43]]
# Set color for MACE and Espresso results
MACE_color = 'blue'
Espr_color = 'gold'

N = 6 # NxN supercell
index=0 #index within axes
for m in ["Mo","W"]:
    for c in ["S","Se"]:

        ax = fig.add_axes(ax_rects[index]); 
        dosax = fig.add_axes(dosax_rects[index]);      
        emin = -25;
        emax = 550;

        prim_opt=mx2(formula=f'{m}{c}2',kind='2H',a=lat_param[f'a_{m}{c}2'],
                     thickness=lat_param[f't_{m}{c}2'],vacuum=8,size=(1,1,1));

        bs_MACE, dos_MACE = get_phonon_bandstructure(prim_opt,N,calc,0.01,f'{m}{c}2_phonon_MACE',read=True)
        plot_phonon_bandstructure(ax,dosax,bs_MACE,dos_MACE,emin,emax,MACE_color,'MACE')

        #Read Espresso phonon calculation 
        bs_Espresso, dos_Espresso = get_phonon_bandstructure(prim_opt,N,calc_espr,0.08,f'{m}{c}2_phonon_espresso',read=True)
        plot_phonon_bandstructure(ax,dosax,bs_Espresso,dos_Espresso,emin,emax,Espr_color,'Espresso')

        # DOS axes customization
        dosax.set_xlabel("DOS",fontsize=fontsize);
        dosax.set_ylim(emin, emax);
        dosax.set_yticks([])
        dosax.set_xticks([])
        
        #Get Mean Absolute Error and Maximum Absolute Error between frequencies over the whole BZ space
        #including all branches for omega>10cm-1
        flattened_bs_MACE = bs_MACE.energies.flatten()
        flattened_bs_Espresso = bs_Espresso.energies.flatten()
        error_args = np.where(flattened_bs_Espresso>10)
        freq_diff = abs(flattened_bs_MACE[error_args]-flattened_bs_Espresso[error_args])
        mae = np.mean(freq_diff)
        max_diff = np.max(freq_diff)
        
        #Define table for showing errors
        stats = {"MAE d$\mathrm{\omega}$:": mae,"MAX d$\mathrm{\omega}$:": max_diff}
        #stats = {"MAE: ": mae,"MAX: ": max_diff}
        my_stats = [stats[x] for x in stats]
        my_labels = [x for x in stats]
        cellText = [[x,f'{stats[x]:0.2f} cm$^{{-1}}$'] for x in stats]      
        ystart = 0.88
        xstart = 0.007
        tab = ax.table(cellText=cellText,bbox=[xstart,ystart,0.35,0.11],cellLoc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(fontsize)
        tab.auto_set_column_width((0,1))
        # Reduce border width
        for key, cell in tab.get_celld().items():
            cell.set_linewidth(0.0) 
        # Add legend for Espresso and MACE
        ax.legend(frameon=False,loc='upper right')

        # Text of (a), (b), (c), (d) for figure
        ax.text(-0.55,0.3,f'({string.ascii_lowercase[index]})',fontsize=fontsize+7,fontweight='extra bold')
        index+=1

#Save the figure
plt.tight_layout()
plt.savefig('phonon_dispersions.png',dpi=300,facecolor='white',
            transparent=False,bbox_inches='tight',pad_inches=0.01)
plt.show()
