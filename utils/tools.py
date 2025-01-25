from scipy.interpolate import LinearNDInterpolator
from ase.build.surface import mx2
from mace.calculators import MACECalculator
import numpy as np

calc = MACECalculator(model_paths="../training/MoWSSe_MACE_201_run-123.model", 
                      device="cuda", default_dtype="float64")

a_optB88_QE = {
    (1, 0): 3.190902132,
    (0, 0): 3.186305534,
    (1, 1): 3.322874304,
    (0, 1): 3.322339468
}

t_optB88_QE = {
    (1, 0): 3.1570497014,
    (0, 0): 3.140170013,
    (1, 1): 3.367178811,
    (0, 1): 3.3492559072
}

# Prepare data for interpolation
xd = np.array([0, 1, 0, 1])
yd = np.array([0, 0, 1, 1])
ad = np.array([a_optB88_QE[(x, y)] for x, y in zip(xd, yd)])
td = np.array([t_optB88_QE[(x, y)] for x, y in zip(xd, yd)])

# Create LinearNDInterpolator instances
a_xy = LinearNDInterpolator(list(zip(xd, yd)), ad)
t_xy = LinearNDInterpolator(list(zip(xd, yd)), td)

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

def passlog():
    pass

def relax_structure(atoms, calc, log_file, fmax=0.001):
    """Relax the given atomic structure inplace and return its energy."""
    atoms.calc = calc
    dyn = BFGS(atoms)
    dyn.log = log_file
    dyn.run(fmax=fmax)

def rgb_to_cmyk(r, g, b):
    # Convert RGB values to range of 0-1
    r, g, b = r/255.0, g/255.0, b/255.0

    # Find the maximum value of RGB values
    max_value = max(r, g, b)

    # If max_value is 0, return 0, 0, 0, 1
    if max_value == 0:
        return 0, 0, 0, 1

    # Calculate the K value
    k = 1 - max_value

    # Calculate the C, M, and Y values
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    # Return the CMYK values
    return c, m, y, k

def cmyk_to_rgb(c, m, y, k):
    # Calculate the RGB values
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)

    # Round the RGB values and return them as integers
    return int(round(r)), int(round(g)), int(round(b))


def get_color(x,y):
    
    #datapoints = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] #1-x,x,2-2y,2y
    
    valuepoints = np.array([[0.8, 0.8, 0.0, 0.2],  #Mo
                            [0.0, 0.0, 0.8, 0.2],  #W
                            [0.0, 0.8, 0.0, 0.2],  #S
                            [0.8, 0.0, 0.0, 0.2]]) #Se     in percentage
    
    cmyk_color = (x*valuepoints[0] + (1-x)*valuepoints[1] + y*valuepoints[2] + (1-y)*valuepoints[3])/2
    
    return tuple(np.array(cmyk_to_rgb(*cmyk_color))/255)