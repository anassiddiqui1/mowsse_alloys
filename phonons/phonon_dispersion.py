import numpy as np;
import string
from mace.calculators import MACECalculator;

from ase.phonons import Phonons;
from ase.optimize import BFGS;
from ase.build.surface import mx2;
import matplotlib.pyplot as plt;
from ase.calculators.espresso import Espresso

import warnings
warnings.filterwarnings('ignore')
fontsize = 8.5
#fontsize = 12
plt.rcParams['font.size'] = fontsize


def passlog():
    pass

a_MoS2 = 3.186305534
t_MoS2 = 3.140170013

a_WS2 = 3.190902132
t_WS2 = 3.1570497014

a_MoSe2 = 3.322339468
t_MoSe2 = 3.3492559072

a_WSe2 = 3.322874304
t_WSe2 = 3.367178811

lat_param = {'a_MoS2': a_MoS2,'t_MoS2': t_MoS2,'a_WS2' : a_WS2, 't_WS2' : t_WS2,
            'a_MoSe2': a_MoSe2,'t_MoSe2': t_MoSe2,'a_WSe2' : a_WSe2, 't_WSe2' : t_WSe2}

calc=MACECalculator(model_paths="../training/MoWSSe_MACE_201_run-123.model", device="cuda", default_dtype="float64")
calc1 = Espresso()

MACE_color = 'blue'
Espr_color = 'gold'
i=0

#fig = plt.figure(figsize=(11,10))
fig = plt.figure(figsize=(9.5,7))

ax_rects = [[0.07,0.52,0.33,0.43],
           [0.56,0.52,0.33,0.43],
           [0.07,0.04,0.33,0.43],
           [0.56,0.04,0.33,0.43]]

dosax_rects = [[0.41,0.52,0.08,0.43],
           [0.9,0.52,0.08,0.43],
           [0.41,0.04,0.08,0.43],
           [0.9,0.04,0.08,0.43]]


for m in ["Mo","W"]:
    for c in ["S","Se"]:

        prim_opt=mx2(formula=f'{m}{c}2',kind='2H',a=lat_param[f'a_{m}{c}2'],
                     thickness=lat_param[f't_{m}{c}2'],vacuum=8,size=(1,1,1));

        #Relax the structure
        prim_opt.calc=calc;
        dyn=BFGS(prim_opt);
        dyn.log = passlog
        
        dyn.run(fmax=0.001);

        #MACE phonon calculation
        N=6;
        ph_MACE = Phonons(prim_opt,calc,supercell=(N, N, 1), delta=0.01, name=f'{m}{c}2_phonon_MACE'); 
#         ph_MACE.clean()
#         ph_MACE.run();
        ph_MACE.read(acoustic=True)
#         ph_MACE.clean()
        path = prim_opt.cell.bandpath('GMKG', npoints=150);
        bs_MACE = ph_MACE.get_band_structure(path,verbose=False); 
        cm1 = 8065.54011;

        #Read Espresso phonon calculation 
        ph_Espresso = Phonons(prim_opt,calc1,supercell=(N, N, 1), delta=0.08, name=f'{m}{c}2_phonon_espresso')
        ph_Espresso.read(acoustic=True)
        bs_Espresso = ph_Espresso.get_band_structure(path,verbose=False)


        #ax = fig.add_axes([.12, .07, .67, .85]);
        ax = fig.add_axes(ax_rects[i]);
        

        #Get DOS for MACE and Espresso
        dos_MACE = ph_MACE.get_dos(kpts=(60, 60, 1)).sample_grid(npts=1000, width=2e-4);
        dos_Espresso = ph_Espresso.get_dos(kpts=(60, 60, 1)).sample_grid(npts=1000, width=2e-4)

        emin = -25;
        emax = 550;
        bs_MACE._energies *= cm1;
        bs_Espresso._energies *= cm1;

        bs_MACE.plot(ax=ax, emin=emin, emax=emax, color=MACE_color, ylabel=r"$\mathrm{\omega}$ (cm$^{-1}$)",label='MACE');
        bs_Espresso.plot(ax=ax, emin=emin, emax=emax, color=Espr_color, ylabel=r"$\mathrm{\omega}$ (cm$^{-1}$)",label='Espresso',linestyle='--')

        #Define axes for DOS
        dosax = fig.add_axes(dosax_rects[i]);
        dosax.plot(dos_MACE.get_weights(), dos_MACE.get_energies()*cm1,color=MACE_color) 
        dosax.plot(dos_Espresso.get_weights(), dos_Espresso.get_energies()*cm1,color=Espr_color,linestyle='--')
        dosax.set_xlabel("DOS",fontsize=fontsize);
        dosax.set_ylim(emin, emax);
        dosax.set_yticks([])
        dosax.set_xticks([])
        
        #Get Mean Absolute Error and Maximum Absolute Error between frequencies over the whole BZ space
        #including all branches
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
    
        ax.legend(frameon=False,loc='upper right')
        ax.text(-0.55,0.3,f'({string.ascii_lowercase[i]})',fontsize=fontsize+7,fontweight='extra bold')
        i+=1
plt.tight_layout()


#Save the figure
plt.savefig('phonon_dispersions.png',dpi=300,facecolor='white',
            transparent=False,bbox_inches='tight',pad_inches=0.01)
plt.show()
