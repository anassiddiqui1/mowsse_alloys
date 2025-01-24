from ase.io import read,write,Trajectory
from glob import glob
from ase import Atoms
from mace.calculators import MACECalculator
from ase.build.surface import mx2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.interpolate import interp2d
from ase.visualize import view
from ase.optimize import BFGS
from ase.geometry.analysis import Analysis
from ase.data import covalent_radii

calc = MACECalculator(model_paths="../training/MoWSSe_MACE_new_version_run-123.model", 
                      device="cuda", default_dtype="float64")

def passlog():
    pass

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

def make_alloy(x,y,N):
    '''
    Make MoWSSe alloy having formula Mo(1-x)W(x)S(2-2y)Se(2y)
    '''
    prim_cell = mx2(formula='MoS2',kind='2H',a=a_xy(x,y),thickness=t_xy(x,y),vacuum=8,size=(1,1,1))
    atoms = prim_cell*(N,N,1)
    nflip_x = int(len(atoms)/3*x)
    flip_x = np.random.choice(range(0,len(atoms),3),nflip_x,replace=False)
    for f in flip_x:
        atoms.symbols[f] = 'W'

    #Flip S with Se
    nflip_y = int(2*len(atoms)/3*y)
    flip_y = np.random.choice([i for i in range(len(atoms)) if i%3!=0],nflip_y,
                              replace=False)
    for f in flip_y:
        atoms.symbols[f] = 'Se'
    return atoms

foldername = 'binding_energy_data'
N = 12
nsamples = 50

##Get binding energy for W-W
x = 0.2
y = 0.0

## Relax MoS2 12x12 cell
mos2_12x12 =mx2(formula='MoS2',kind='2H',a=a_xy(x,y),thickness=t_xy(x,y),vacuum=8,size=(N,N,1))
mos2_12x12.calc = calc
dyn = BFGS(mos2_12x12)
dyn.log = passlog
dyn.run(fmax=0.001)
mos2_energy = mos2_12x12.get_potential_energy()
##

### Relax MoS2 12x12 cell with one Mo replaced by W
mows2_12x12 = mos2_12x12.copy()
mows2_12x12.symbols[30] = 'W'
mows2_12x12.calc = calc
dyn = BFGS(mows2_12x12)
dyn.log = passlog
dyn.run(fmax=0.001)
mows2_energy = mows2_12x12.get_potential_energy()
##

nsamples = 50
WWpairs = np.zeros(nsamples)
energies = np.zeros(nsamples)

tout = Trajectory(f'{foldername}/MoWS2_nn.traj',"w")

for i in range(nsamples):

    atoms = make_alloy(x,y,N)
    tout.write(atoms)
    n_W = atoms.symbols.count('W')
    #Calculate number of W-W bonds
    ana = Analysis(atoms)
    Wpairs = ana.get_bonds('W','W',unique=True)[0]
    total_Wpairs = len(Wpairs)
    print(f'Running Sample {i}')
    atoms.calc = calc
    dyn=BFGS(atoms)
    dyn.log = passlog
    dyn.run(fmax=0.001)
    atoms_energy = atoms.get_potential_energy()

    #Calculate binding energy
    binding_e = atoms_energy - n_W*mows2_energy + (n_W-1)*mos2_energy
    WWpairs[i] = total_Wpairs
    energies[i] = binding_e
np.savez(f'{foldername}/MoWS2_nn.npz',nns=WWpairs,energies=energies)

###Get binding energy for Se-Se
covalent_radii[34] = 1.7
x = 0.0
y = 0.2

### Relax MoS2 12x12 cell with one S replaced by Se
mosse_12x12 = mos2_12x12.copy()
mosse_12x12.symbols[202] = 'Se'
mosse_12x12.calc = calc
dyn = BFGS(mosse_12x12)
dyn.log = passlog
dyn.run(fmax=0.001)
mosse_energy = mosse_12x12.get_potential_energy()
##

n_Se_pairs = np.zeros(nsamples)
n_Se_z_pairs = np.zeros(nsamples)
n_Se_plane_pairs = np.zeros(nsamples)
energies = np.zeros(nsamples)

tout = Trajectory(f'{foldername}/MoSSe_nn.traj',"w")


for i in range(nsamples):

    print(f'Running Sample {i}')

    atoms = make_alloy(x,y,N)
    tout.write(atoms)
    n_Se = atoms.symbols.count('Se')

    #Calculate number of Se-Se bonds
    ana = Analysis(atoms)
    Se_pairs = ana.get_bonds('Se','Se',unique=True)[0]
    total_Se_pairs = len(Se_pairs)

    z_Se_pairs = 0
    plane_Se_pairs = 0
    for Se_pair in Se_pairs:
        if abs(Se_pair[0]-Se_pair[1]) == 1:
            z_Se_pairs +=1
    plane_Se_pairs = total_Se_pairs - z_Se_pairs
    #print(plane_Se_pairs,z_Se_pairs,total_Se_pairs)
    atoms.calc = calc
    dyn=BFGS(atoms)
    dyn.log = passlog
    dyn.run(fmax=0.001)
    atoms_energy = atoms.get_potential_energy()

    #Calculate binding energy
    binding_e = atoms_energy - n_Se*mosse_energy + (n_Se-1)*mos2_energy
    n_Se_pairs[i] = total_Se_pairs
    n_Se_z_pairs[i] = z_Se_pairs
    n_Se_plane_pairs[i] = plane_Se_pairs
    energies[i] = binding_e

np.savez(f'{foldername}/MoSSe_nn.npz',nns=n_Se_pairs,
         nns_vertical = n_Se_z_pairs,nns_lateral = n_Se_plane_pairs,
         energies = energies)
