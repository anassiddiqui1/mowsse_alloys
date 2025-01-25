import matplotlib.pyplot as plt
from ase.io import read
from glob import glob
import numpy as np
from os import chdir
from scipy.interpolate import interp2d
import sys
from pathlib import Path

repo_path = Path("../utils")
sys.path.append(str(repo_path))
from tools import calc

FONTSIZE = 8.3
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams["figure.figsize"] = (4.75,3.5)

#Working directory 
chdir('test_data/')

#Load atomic energies
E_Mo = read('Mo.pwo').get_potential_energy()
E_W = read('W.pwo').get_potential_energy()
E_S = read('S.pwo').get_potential_energy()
E_Se = read('Se.pwo').get_potential_energy()
###Get formation energies for parent layers at 100K for scaling the total energies
f_energy_QE = {}
atoms_MoS2 = read('x0.000y0.000s1.000T0100.pwo')
f_energy_QE[0,0] = atoms_MoS2.get_potential_energy()-36*E_Mo-72*E_S
atoms_MoSe2 = read('x0.000y1.000s1.000T0100.pwo')
f_energy_QE[0,1] = atoms_MoSe2.get_potential_energy()-36*E_Mo-72*E_Se
atoms_WS2 = read('x1.000y0.000s1.000T0100.pwo')
f_energy_QE[1,0] = atoms_WS2.get_potential_energy()-36*E_W-72*E_S
atoms_WSe2 = read('x1.000y1.000s1.000T0100.pwo')
f_energy_QE[1,1] = atoms_WSe2.get_potential_energy()-36*E_W-72*E_Se
xd = np.array([0,1,0,1])
yd = np.array([0,0,1,1])


##Interpolation function for alloys formation energy
E_QE = [0]*len(xd)
for q in range(len(xd)):
    E_QE[q] = f_energy_QE[xd[q],yd[q]]
E_xy_QE = interp2d(xd,yd,E_QE)

# x and y compositions within test dataset
xs = np.linspace(0,1,7)
ys = np.linspace(0,1,7)

###Arrays to store forces and energies for both Espresso (target) and MACE (predict)
nfiles = len(glob(f'*.pwo'))
target_energies = np.zeros(nfiles)
predicted_energies = np.zeros(nfiles)
target_forces = np.zeros((nfiles,108,3))
predicted_forces = np.zeros((nfiles,108,3))

files = glob(f'x*y*.pwo')
for j,file in enumerate(files):
    atoms = read(file)
    n_Mo = atoms.symbols.count('Mo')
    n_W = atoms.symbols.count('W')
    n_S = atoms.symbols.count('S')
    n_Se = atoms.symbols.count('Se')
    xp = (n_W)/(n_Mo+n_W)
    yp = (n_Se)/(n_S+n_Se)
    total_atomic_energy = n_Mo*E_Mo + n_W*E_W + n_S*E_S + n_Se*E_Se
    target_energies[j] = atoms.get_potential_energy() - (total_atomic_energy + E_xy_QE(xp,yp))
    target_forces[j] = atoms.get_forces()
    atoms.calc = calc
    predicted_energies[j] = atoms.get_potential_energy() - total_atomic_energy -E_xy_QE(xp,yp)
    predicted_forces[j] = atoms.get_forces()
#Change directory back to parent
chdir('../')

###Get MAE,MAX(F and E) and RMS (F)  
rms_fd = np.zeros(len(target_forces))

## Calculate rmse in forces for each configuration between DFT and MLIP
for i in range(len(target_forces)):
     rms_fd[i] = np.sqrt(np.mean((target_forces[i].flatten()-predicted_forces[i].flatten())**2))

# Calculate MAE and MAX for energies and forces
mae_e = np.mean(np.abs(target_energies-predicted_energies))
mae_f = np.mean(np.abs(target_forces.flatten()-predicted_forces.flatten()))
max_e = np.max(np.abs(target_energies-predicted_energies))
max_f = np.max(np.abs(target_forces.flatten()-predicted_forces.flatten()))

stats = {"MAE dE (eV)  ": mae_e,
        "MAE dF (eV/A)": mae_f,
        "MAX dE (eV)  ": max_e,
        "MAX dF (eV/A)": max_f}

im = plt.scatter(target_energies,predicted_energies,c=rms_fd) 
#Plot colorbar
cb = plt.colorbar(im)
cb.set_label(r'RMS Force error (eV/$\AA$)')

# y=x line
line_range = np.linspace(min(target_energies)-.2,max(target_energies)+.2)
plt.plot(line_range,line_range,'k-', alpha=0.75, zorder=0)

#Set x and y axis limits
plt.xlim(min(target_energies)-.4,max(target_energies)+.4)
plt.ylim(min(predicted_energies)-.4,max(predicted_energies)+.4)

#Stats table
my_stats = [stats[x] for x in stats]
my_labels = [x for x in stats]
cellText = [[f'{x:0.4f}'] for x in my_stats]
tab = plt.table(rowLabels=my_labels,cellText=cellText,
                bbox=[0.34,0.76,0.15,0.22],cellLoc='center')

tab.auto_set_column_width(col=[0,1])
# Reduce border width
for key, cell in tab.get_celld().items():
    cell.set_linewidth(0.3)  # Set the desired border width

#Assign labels
plt.xlabel('Energy from DFT (eV)')
plt.ylabel('Energy from MLIP (eV)')
plt.tight_layout()

plt.savefig('target_vs_predict.png',dpi=300,facecolor='white',transparent=False,bbox_inches='tight',pad_inches=0.01)

plt.show()
