# Load standard libraries
import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py

# Find relative local paths to stuff
path1p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
path2p = os.path.dirname(path1p)
pwd_lib = os.path.join(path1p, "lib/")
pwd_rez = os.path.join(os.path.join(path2p, "data/"), "sim-ds-mat")

print(pwd_lib)

# Set paths
sys.path.append(pwd_lib)

# Load user libraries
from matlab.matlab_lib import loadmat

'''
  TODO:
    1) Load data from HDF5
    2) Save data as temporary .mat
    3) Run matlab NIfTy to get TE
    4) Load NIfTy TE output from .mat
    5) Save results as HDF5
'''

def nifty_wrapper(src_path_h5):
    # 1) Load data from HDF5
    print("Reading source data from", src_path_h5)
    src_file_h5 = h5py.File(src_path_h5, "r")
    src_data = np.copy(src_file_h5['data'])
    src_file_h5.close()
    
    print("read data from H5 : ", src_data.shape, type(src_data))
    
    # 2) Save data as temporary .mat
    src_path_mat = os.path.join(pwd_rez, "source_selftest_rand.mat")
    print("Converting data to matlab file", src_path_mat)
    scipy.io.savemat(src_path_mat, {"data" : src_data})

    # 3) Run matlab NIfTy to get TE
    print("Started running matlab")
    #strFun = 'nifty_wrapper("' + path2p + '/")'
    
    action1 = 'core_path = "' + path2p + '/";'
    action2 = 'addpath(char(core_path + "codes/lib/nifty_wrapper/"));'
    action3 = 'run("nifty_wrapper.m");'
    action_sum = action1 + action2 + action3 + "exit;"
    
    print("..Action:", action_sum)
    subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", action_sum])
    # '"run(' +"'test_nifty_alyosha.m');exit;" + '"'

    # 4) Load NIfTy TE output from .mat
    rez_path_mat = os.path.join(pwd_rez, "results_selftest_rand.mat")
    print("Loading matlab results file", rez_path_mat)
    rez_data = loadmat(rez_path_mat)
    
    # 5) Save results as HDF5
    rez_path_h5 = os.path.join(pwd_rez, "results_selftest_rand.h5")
    print("Writing results data to", rez_path_h5)
    rez_file_h5 = h5py.File(rez_path_h5, "w")
    rez_file_h5['results'] = rez_data['results']
    rez_file_h5.close()
    
    return rez_path_h5
