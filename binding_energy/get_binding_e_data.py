from ase.io import Trajectory
from ase.optimize import BFGS
from ase.geometry.analysis import Analysis
from ase.data import covalent_radii
import numpy as np
import sys
from pathlib import Path
repo_path = Path("../utils")
sys.path.append(str(repo_path))
from tools import mx2, a_xy, t_xy, calc, passlog, make_alloy,relax_structure

# Configuration
FOLDER_NAME = 'binding_energy_data'  # Folder to store numpy arrays
N = 12  # NxN supercell
NSAMPLES = 50 # Number of samples

def relax_structure(atoms, calc, log_file, fmax=0.001):
    """Relax the given atomic structure inplace and return its energy."""
    atoms.calc = calc
    dyn = BFGS(atoms)
    dyn.log = log_file
    dyn.run(fmax=fmax)
    return atoms.get_potential_energy()

def calculate_binding_energy(atoms_energy, element_energy, base_energy, n_elements):
    """Calculate the binding energy for the given structure."""
    return atoms_energy - n_elements * element_energy + (n_elements - 1) * base_energy

def analyze_structure(atoms, element):
    """Analyze the structure and return the number of element-element bonds."""
    ana = Analysis(atoms)
    pairs = ana.get_bonds(element, element, unique=True)[0]
    total_pairs = len(pairs)
    z_pairs = sum(1 for pair in pairs if abs(pair[0] - pair[1]) == 1)
    plane_pairs = total_pairs - z_pairs
    return total_pairs, z_pairs, plane_pairs

def process_samples(x, y, calc, element, reference_energy, base_energy, output_file, traj_file):
    """Process samples and save bond analysis and binding energies."""
    n_pairs = np.zeros(NSAMPLES)
    n_vertical_pairs = np.zeros(NSAMPLES)
    n_lateral_pairs = np.zeros(NSAMPLES)
    energies = np.zeros(NSAMPLES)
    tout = Trajectory(traj_file, "w")

    for i in range(NSAMPLES):
        print(f'Running Sample {i+1}')
        atoms = make_alloy(x, y, N)
        tout.write(atoms)
        n_elements = atoms.symbols.count(element)

        # Analyze bonds
        total_pairs, z_pairs, plane_pairs = analyze_structure(atoms, element)
        n_pairs[i] = total_pairs
        n_vertical_pairs[i] = z_pairs
        n_lateral_pairs[i] = plane_pairs

        # Relax structure and calculate binding energy
        atoms_relaxed_energy = relax_structure(atoms,calc, passlog)
        binding_energy = calculate_binding_energy(atoms_relaxed_energy, reference_energy, base_energy, n_elements)
        energies[i] = binding_energy

    np.savez(output_file, nns=n_pairs, nns_vertical=n_vertical_pairs, nns_lateral=n_lateral_pairs, energies=energies)

## Relax MoS2 supercell cell
mos2_supercell =mx2(formula='MoS2',kind='2H',a=a_xy(0,0),thickness=t_xy(0,0),vacuum=8,size=(N,N,1))
mos2_supercell.calc = calc
mos2_energy = relax_structure(mos2_supercell, calc, passlog)

### Relax MoS2 supercell cell with one Mo replaced by W
mows2_supercell = mos2_supercell.copy()
mows2_supercell.symbols[0] = 'W'
mows2_supercell.calc = calc
mows2_energy = relax_structure(mows2_supercell, calc, passlog)

# Process W-W samples
process_samples(
    x=0.2,
    y=0.0,
    calc=calc,
    element='W',
    reference_energy=mows2_energy,
    base_energy=mos2_energy,
    output_file=f'{FOLDER_NAME}/MoWS2_nn.npz',
    traj_file=f'{FOLDER_NAME}/MoWS2_nn.traj'
)

# Update covalent radii for Se
covalent_radii[34] = 1.7

### Relax MoS2 supercell cell with one Mo replaced by W
mosse_supercell = mos2_supercell.copy()
mosse_supercell.symbols[1] = 'Se'
mosse_supercell.calc = calc
mosse_energy = relax_structure(mosse_supercell, calc, passlog)

# Process Se-Se samples
process_samples(
    x=0.0,
    y=0.2,
    calc=calc,
    element='Se',
    reference_energy=mosse_energy,
    base_energy=mos2_energy,
    output_file=f'{FOLDER_NAME}/MoSSe_nn.npz',
    traj_file=f'{FOLDER_NAME}/MoSSe_nn.traj'
)
