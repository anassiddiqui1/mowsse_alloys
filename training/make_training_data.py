from ase.calculators.kim.kim import KIM
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
from ase.build.surface import mx2
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from ase.io import Trajectory
import numpy as np
import sys
from pathlib import Path
repo_path = Path("../utils")
sys.path.append(str(repo_path))
from tools import a_xy, t_xy, make_alloy,passlog

#Make 6x6 MoS2 supercell
atoms = make_alloy(0,0,6)
#Load the Stillinger-Weber potential for MoS2
calc = KIM("SW_MX2_WenShirodkarPlechac_2017_MoS__MO_201919462778_001")
atoms.calc = calc
# ExpCellFilter to optimise just a and b lengths (fixed angles and cell height)
dyn = BFGS(ExpCellFilter(atoms,mask=[True,True,False,False,False,False]))
dyn.log = passlog
dyn.run(fmax=0.0001)

# Setup details
xs = np.linspace(0,1,7)  #W compositions in alloy
ys = np.linspace(0,1,7)  #Se compositions in alloy
temps = np.linspace(100,1100,6) # MD temperatures from 100 to 1100 in steps of 200K
strains = np.linspace(1.02,0.98,5) # Strains from +2% down to -2% in steps of 1%

tout = Trajectory(f"MoWSSe_SWPot.traj","w")  #Trajectory to store configurations for training data

for x in xs:
    for y in ys:
        for strain in strains:
            # Create and scale the MD cell
            atoms_md = atoms.copy()
            atoms_md.set_cell(atoms_md.cell*np.array([strain,strain,1.0]),scale_atoms=True)
            a_SW = atoms_md.cell[0,0]/6
            t_SW = atoms_md.positions[1,2]-atoms_md.positions[2,2]
            atoms_md.calc = calc
            name = f'md_s{strain:05.3f}'
            # Create Langevin dynamics ASE driver
            dyn = Langevin(atoms_md, timestep=5.0 * units.fs, temperature_K=temps[0], friction=0.01,
                 trajectory=None, logfile=None) 
            # Initial velocities
            MaxwellBoltzmannDistribution(atoms_md, temperature_K=temps[0])
            Stationary(atoms_md)

            # Temperatures from 100 to 1100 in steps of 200K
            for temp in temps:

                atoms_xy = atoms_md.copy()

                #Flip Mo with W
                nflip_x = int(len(atoms_xy)/3*x)
                flip_x = np.random.choice(range(0,len(atoms_xy),3),nflip_x,replace=False)
                for f in flip_x:
                    atoms_xy.symbols[f] = 'W'
                #Flip S with Se
                nflip_y = int(2*len(atoms_xy)/3*y)
                flip_y = np.random.choice([i for i in range(len(atoms_xy)) if i%3!=0],nflip_y,
                                          replace=False)
                for f in flip_y:
                    atoms_xy.symbols[f] = 'Se'

                #Scale cell with atoms
                cell_scale = np.array([strain*a_xy(x,y)[0]/a_SW,
                                       strain*a_xy(x,y)[0]/a_SW,
                                       t_xy(x,y)[0]/t_SW])
                atoms_xy.set_cell(atoms_xy.cell*cell_scale,scale_atoms=True)
                tout.write(atoms_xy)

                #Log MD step
                print(f'{x:5.3f} {y:5.3f} {strain:5.3f} {temp:4.0f} {a_xy(x,y)[0]:10.8f} {t_xy(x,y)[0]:10.8f}')
                dyn.temperature_K = temp
                dyn.run(1000)

tout.close()
