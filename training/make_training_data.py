from ase.calculators.kim.kim import KIM
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
from ase.build.surface import mx2
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.vibrations import Vibrations
from ase import units
from ase.io import Trajectory
import sys
from scipy.interpolate import interp2d

atoms = tmd_66.copy()*(1,1,1)
calc = KIM("SW_MX2_WenShirodkarPlechac_2017_MoS__MO_201919462778_001")
atoms.calc = calc

energy = atoms.get_potential_energy()
print("Unoptimised Potential energy: \n{} eV".format(energy))

# ExpCellFilter to optimise just a and b lengths (fixed angles and cell height)
dyn = BFGS(ExpCellFilter(atoms,mask=[True,True,False,False,False,False]))
dyn.run(fmax=0.0001)

# Save and print result
atoms_opt = atoms.copy()
print("Optimised Potential energy: \n{} eV".format(energy))
print('Optimized cell:')
for i in range(0,3):
    for j in range(0,3):
        print(f'{atoms_opt.cell[i,j]:16.12f} ',end="\n" if j==2 else "")
        #
a_SWPot = atoms_opt.cell[0,0]/6
t_SWPot = atoms_opt.positions[1,2]-atoms_opt.positions[2,2]


a_optB88_QE = {}
a_optB88_QE[1,0] = 3.190902132
a_optB88_QE[0,0] = 3.186305534
a_optB88_QE[1,1] = 3.322874304
a_optB88_QE[0,1] = 3.322339468

t_optB88_QE = {}
t_optB88_QE[1,0] = 3.1570497014
t_optB88_QE[0,0] = 3.140170013
t_optB88_QE[1,1] = 3.367178811
t_optB88_QE[0,1] = 3.3492559072

# Create linear interpolation
xd = np.array([0,1,0,1])
yd = np.array([0,0,1,1])
ad = [0]*len(xd)
td = [0]*len(xd)
for q in range(len(xd)):
    ad[q] = a_optB88_QE[xd[q],yd[q]]
    td[q] = t_optB88_QE[xd[q],yd[q]]
a_xy = interp2d(xd,yd,ad)
t_xy = interp2d(xd,yd,td)

xs = np.linspace(0,1,7)
ys = np.linspace(0,1,7)
temps = np.linspace(100,1100,6)
strains = np.linspace(1.02,0.98,5)

tout = Trajectory(f"MoWSSe_SWPot.traj","w")
#Loop over W percentage
for x in xs:
    #Loop over Se percentage
    for y in ys:
        # Strains from +2% down to -2% in steps of 1%
        for strain in strains:

            # Create and scale the MD cell
            atoms_md = atoms_opt.copy()
            atoms_md.set_cell(atoms_md.cell*np.array([strain,strain,1.0]),scale_atoms=True)
            a_SW = atoms_md.cell[0,0]/6
            t_SW = atoms_md.positions[1,2]-atoms_opt.positions[2,2]
            atoms_md.calc = calc
            name = f'md_s{strain:05.3f}'
            # Create Langevin dynamics ASE driver
            dyn = Langevin(atoms_md, timestep=5.0 * units.fs, temperature_K=temps[0], friction=0.01,
                 trajectory=None, logfile=None) 
            # Initial velocities
            MaxwellBoltzmannDistribution(atoms_md, temperature_K=temps[0])
            Stationary(atoms_md)
            # Get forces on initial frame, before dynamics starts
            atoms_md.get_forces()

            # Temperatures from 100 to 1100 in 10 steps of 200K
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

                print(f'{x:5.3f} {y:5.3f} {strain:5.3f} {temp:4.0f} {a_xy(x,y)[0]:10.8f} {t_xy(x,y)[0]:10.8f}')

                #print(name,i,atoms_md.get_potential_energy())
                dyn.temperature_K = temp
                dyn.run(1000)
                #print(atoms_xy)
tout.close()
