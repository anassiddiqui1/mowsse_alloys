from ase.build.surface import mx2
import numpy as np
from ase.vibrations import Vibrations
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from scipy.interpolate import interp2d
from os import chdir

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


calc=MACECalculator(model_paths="../training/MoWSSe_MACE_201_run-123.model", device="cuda", default_dtype="float64")
chdir("vdos_data/")

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

# Create linear interpolation for lattice params
xd = np.array([0,1,0,1])
yd = np.array([0,0,1,1])
ad = [0]*len(xd)
td = [0]*len(xd)
for q in range(len(xd)):
    ad[q] = a_optB88_QE[xd[q],yd[q]]
    td[q] = t_optB88_QE[xd[q],yd[q]]
a_xy = interp2d(xd,yd,ad)
t_xy = interp2d(xd,yd,td)

##x and y ranges for alloy compositions
xs = np.linspace(0,1,5)
ys = np.linspace(0,1,5)
N= 6 #Define NxN cell
for x in xs:
    for y in ys:

        dos_sum = np.zeros(501)

        #Only need one config for pure materials, 15 for alloys
        if (x,y) in [(0,0),(0,1),(1,0),(1,1)]:
            nconfigs = 1
        else:
            nconfigs = 15

        for  j in range(nconfigs):

            #Make alloy and relax it
            atoms = make_alloy(x,y,N)
            atoms.calc = calc
            dyn=BFGS(atoms)
            dyn.run(fmax=0.001)
            #Run vibrations calculation 
            vib = Vibrations(atoms,name=f'x{x:5.3f}y{y:5.3f}sample{j}',delta=0.02)
            vib.clean()
            vib.run()
            vib.summary()
            vib.write_dos(out=vib.name+"_dos.dat",start=0,end=500)
            vib.clean()
            dos = np.loadtxt(vib.name+"_dos.dat")
            #plt.plot(dos[:,0],dos[:,1])
            dos_sum = dos_sum + dos[:,1]
        #Save total density of states
        np.savetxt(f'x{x:5.3f}y{y:5.3f}dossum.dat',dos_sum)
