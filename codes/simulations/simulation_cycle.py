# Load standard libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Find relative local paths to stuff
path1p = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path2p = os.path.dirname(path1p)
pwd_lib = os.path.join(path1p, "lib/")
pwd_rez = os.path.join(os.path.join(path2p, "data/"), "sim_ds_h5")

#######################
# Generate Data
#######################
    
N_NODE = 5
N_DATA = 4000

# Generate data that is cycled backwards for every new node
# Thus the node with highest index should appear to have the "earliest" version of the same data
src_data = np.zeros((N_NODE, N_DATA))
src_data[0] = np.random.normal(0, 1, N_DATA)
for i in range(1, N_NODE):
    src_data[i] = np.hstack((src_data[i-1][1:], src_data[i-1][0]))
    
# Have to add some noise to data or IDTxl will blow up :D
src_data += np.random.normal(0, 0.1, N_NODE*N_DATA).reshape((N_NODE, N_DATA))

########################
## Save result as HDF5
########################
src_path_h5 = os.path.join(pwd_rez, "sim_cycle_1.h5")
print("Writing source data to", src_path_h5)
src_file_h5 = h5py.File(src_path_h5, "w")
src_file_h5['data'] = src_data
src_file_h5.close()