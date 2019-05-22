import os, sys
from datetime import datetime
import numpy as np
import scipy.io

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from matlab.matlab_lib import loadmat

def read_lvm(filename):
    print("Reading LVM file", filename, "... ")
    
    # Read file
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Figure out where the headers are
    header_endings = [i for i in range(len(lines)) if "End_of_Header" in lines[i]]
        
    # Read data after the last header ends
    idx_data_start = header_endings[-1] + 2
    data = np.array([line.strip().split('\t') for line in lines[idx_data_start:]]).astype(float)
    channel_idxs = data[:,0].astype(int)
    times = data[:,1]

    # Figure out how many channels there are
    min_channel = np.min(channel_idxs)
    max_channel = np.max(channel_idxs)
    nChannel = max_channel - min_channel + 1
    nData = len(channel_idxs)//nChannel

    # Partition data into 2D array indexed by channel
    data2D = np.zeros((nChannel, nData))
    for i in range(nChannel):
        data2D[i] = times[channel_idxs == i]
        
    print("... done! Data shape read ", data2D.shape)
    return data2D


# Convert "scipy.io.matlab.mio5_params.mat_struct object" to dict
def matstruct2dict(matstruct):
    return {s : [getattr(matstruct, s)] for s in dir(matstruct) if s[0]!='_'}

# Merge 2 dictionaries, given that values of both are lists
def merge_dicts(d_lst):
    d_rez = d_lst[0]
    for i in range(1, len(d_lst)):
        d_rez = {k1 : v1 + d_lst[i][k1] for k1, v1 in d_rez.items()}
    return d_rez

# Read data and behaviour matlab files given containing folder
def read_mat(folderpath):
    # Read MAT file from command line
    print("Reading Yaro data from", folderpath)
    datafilename = os.path.join(folderpath, "data.mat")
    behaviorfilename = os.path.join(folderpath, "behaviorvar.mat")

    data = loadmat(datafilename)['data']
    behavior = loadmat(behaviorfilename)
    
    # Get rid of useless fields in behaviour
    behavior = {k : v for k, v in behavior.items() if k[0] != '_'}
    
    # Convert trials structure to a dictionary
    behaviour['trials'] = merge_dicts([matstruct2dict(obj) for obj in behavior['trials']])
    
    # d_trials = matstruct2dict(behavior['trials'][0])
    # for i in range(1, len(behavior['trials'])):
    #     d_trials = merge_dict(d_trials, matstruct2dict(behavior['trials'][i]))
    # behavior['trials'] = d_trials
    
    return data, behavior


def get_subfolders(folderpath):
    return [_dir for _dir in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, _dir))]


def read_mat_multi(rootpath):
    
    # Get all subfolders, mark them as mice
    mice = get_subfolders(rootpath)
    micedict = {}
    
    # For each mouse, get all subfolders, mark them as days
    for mouse in mice:
        mousepath = os.path.join(rootpath, mouse)
        days = get_subfolders(mousepath)
        micedict[mouse] = {day : {} for day in days}

        # For each day, read mat files
        for day in days:
            daypath = os.path.join(mousepath, day)
            data, behaviour = read_mat(daypath)
            micedict[mouse][day] = {
                'data' : data,
                'behaviour' : behaviour
            }
            
    return micedict
