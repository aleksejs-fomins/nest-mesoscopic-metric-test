# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# Locate results path
rezPath = os.path.join(os.path.join(p2path, 'data/'), 'sim-ds-py')

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
from dyn_sys import DynSys
from corr_lib import crossCorr
from plots.plt_imshow_mat import plotImshowMat


# Set parameters
param = {
    'ALPHA'   : 0.1,  # 1-connectivity strength
    'N_NODE'  : 12,   # Number of variables
    'N_DATA'  : 4000, # Number of timesteps
    'MAG'     : 0,    # Magnitude of input
    'T'       : 20,   # Period of input oscillation
    'STD'     : 0.2   # STD of neuron noise
}

# Create dynamical system
DS1 = DynSys(param)

# Save simulation and metadata
# DS1.save(os.path.join(rezPath, "testDynSys-1.h5"))

#Plot results
DS1.plot(draw=True)


#######################
#  Cross-correlation
#######################

DELAY_MIN = 1
DELAY_MAX = 5

corrMat, corrDelMat = crossCorr(DS1.data, DELAY_MIN, DELAY_MAX, est='corr')
sprMat,  sprDelMat = crossCorr(DS1.data, DELAY_MIN, DELAY_MAX, est='spr')

plotImshowMat(
    [[corrMat, corrDelMat], [sprMat,  sprDelMat]],
    np.array([['Corr', 'Spr'],['CorrDel', 'SprDel']]),
    "Cross-correlation for DynSys",
    lims = [[[-1,1], [0, DELAY_MAX]], [[-1,1], [0, DELAY_MAX]]],
    draw=True
)

#######################
#  Cross-correlation with velocities
#######################

data2 = np.vstack((DS1.data[:, :-1], DS1.data[:, 1:] - DS1.data[:, :-1]))

print(data2.shape)

corrMat, corrDelMat = crossCorr(data2, DELAY_MIN, DELAY_MAX, est='corr')
sprMat,  sprDelMat = crossCorr(data2, DELAY_MIN, DELAY_MAX, est='spr')

plotImshowMat(
    [[corrMat, corrDelMat], [sprMat,  sprDelMat]],
    np.array([['Corr', 'Spr'],['CorrDel', 'SprDel']]),
    "Cross-correlation for DynSys - with velocities",
    lims = [[[-1,1], [0, DELAY_MAX]], [[-1,1], [0, DELAY_MAX]]],
    draw=True
)







plt.show()
