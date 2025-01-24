import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

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

fontsize = 8.3
#fontsize = 4
plt.rcParams['font.size'] = fontsize
plt.rcParams["figure.figsize"] = (4.75,3.75)

xs = np.linspace(0,1,5)
ys = np.linspace(0,1,5)

fig,axs = plt.subplots(5,5)

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        color = get_color(x,y)
        dos = np.loadtxt(f'vdos_data/x{x:5.3f}y{y:5.3f}dossum.dat')
        
        if (x,y) in [(0,0),(0,1),(1,0),(1,1)]:
            nconfigs = 1
        else:
            nconfigs = 15
        ###Plot average density of states
        axs[i][j].plot(dos/nconfigs,color=color)
        
        axs[i][j].set_ylim(-1,56)
        axs[i][j].xaxis.set_minor_locator(MultipleLocator(50))
        axs[i][j].xaxis.set_major_locator(MultipleLocator(200))
        #axs[i][j].set_xlim(0,550)
        #axs[j][i%5].set_title(f'x{x:5.3f}y{y:5.3f}',x=0.2, y=0.85)
        
        #if j!=0:
        axs[i][j].set_yticks([])
        if j==0:
            axs[i][j].set_ylabel(f'x={x}')
            #axs[i][j].yaxis.set_minor_locator(MultipleLocator(10))
        
        if i!=4:
            axs[i][j].tick_params(labelbottom=False)
        
        if i==0:
            axs[i][j].set_xlabel(f'y={y}')
            axs[i][j].xaxis.set_label_position('top') 
            

fig.text(0.5, 0.04, r"$\mathrm{\omega}$ (cm$^{-1}$)", ha='center')
fig.text(0.06, 0.5, 'Density of States (a.u.)', va='center', rotation='vertical')

plt.subplots_adjust(wspace=0.05, hspace=0.15)
# fig.tight_layout(pad=0)

plt.savefig('vibrational_dos_mace.png',dpi=300,facecolor='white',transparent=False,
            bbox_inches='tight',pad_inches=0.02)

plt.show()
