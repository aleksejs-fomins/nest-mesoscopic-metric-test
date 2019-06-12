import numpy as np
import os, sys
import h5py

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from os_lib import getfiles_walk

def parseTE_H5(fname):
    print("Reading file", fname)
    filename = os.path.join(pwd_h5, os.path.join("real_data", fname))
    h5f = h5py.File(filename, "r")
    TE = np.copy(h5f['results']['TE_table'])
    lag = np.copy(h5f['results']['delay_table'])
    p = np.copy(h5f['results']['p_table'])
    h5f.close()
    return (TE, lag, p)


def parseTEfolders(folderpaths):
    
    
    
    for 

    datafilesets = []
    basenamesets = []
    statistics = []

    # GUI: Select videos for training
    pwd_tmp = "./"
    datafilenames = None
    while datafilenames != ['']:
        datafilenames = gui_fnames("IDTXL swipe result files...", directory=pwd_tmp, filter="HDF5 Files (*.h5)")
        if datafilenames != ['']:
            print("Total user files in dataset", len(datafilesets), "is", len(datafilenames))   

            pwd_tmp = os.path.dirname(datafilenames[0])  # Next time choose from parent folder
            datafilesets += [np.array(datafilenames)]
            basenamesets += [np.array([os.path.basename(name) for name in datafilenames])]
            statistics += [getStatistics(basenamesets[-1])]

    print('Selecting files Done, reading...')
    datasets = [np.array([getData(fname) for fname in fnames]) for fnames in datafilesets]