# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(parpath, 'lib/'))

# Locate results path
rezPath = os.path.join(os.path.join(p2path, 'data/')), '')

import dyn_sys

# Set parameters
param = {
    'ALPHA'   : 0.9,  # 1-connectivity strength
    'N_NODE'  : 12,   # Number of variables
    'N_DATA'  : 4000, # Number of timesteps
    'T'       : 100,  # Period of input oscillation
    'STD'     : 0.2   # STD of neuron noise
}

# Create dynamical system
DS1 = DynSys(param)

# Save simulation and metadata
DS1.save('')

#Plot results
DS1.plot()
