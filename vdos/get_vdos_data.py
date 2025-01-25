import numpy as np
from ase.vibrations import Vibrations
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from scipy.interpolate import interp2d
from os import chdir
import sys
from pathlib import Path
repo_path = Path("../utils")
sys.path.append(str(repo_path))
from tools import make_alloy,passlog,relax_structure,calc

chdir("vdos_data/")
##x and y ranges for alloy compositions
xs = np.linspace(0,1,5)
ys = np.linspace(0,1,5)
N= 6 #Define NxN cell
for x in xs:
    for y in ys:
        dos_sum = np.zeros(501)
        #Only need one config for pure materials, averaged over 15 for alloys
        if (x,y) in [(0,0),(0,1),(1,0),(1,1)]:
            nconfigs = 1
        else:
            nconfigs = 15 

        for  j in range(nconfigs):
            #Make alloy and relax it
            atoms = make_alloy(x,y,N)
            relax_structure(atoms,calc,passlog)
            #Run vibrations calculation 
            vib = Vibrations(atoms,name=f'x{x:5.3f}y{y:5.3f}sample{j}',delta=0.01)
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
