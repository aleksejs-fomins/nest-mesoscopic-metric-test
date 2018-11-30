import os, sys
from datetime import datetime
import numpy as np
import scipy.io

def read_lvm(filename):
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
        
    print("Read LVM file", filename, "with data shape", data2D.shape)
    return data2D


def read_mat(folderpath):
    # Read MAT file from command line
    print("Reading Yaro data from", folderpath)
    datafilename = os.path.join(folderpath, "data.mat")
    behaviorfilename = os.path.join(folderpath, "behaviorvar.mat")

    mat1 = scipy.io.loadmat(datafilename)
    mat2 = scipy.io.loadmat(behaviorfilename)

    #d_behaviour = {
        #'trials_1'  : np.array([row[0][0][0][0] for row in mat2['trials']]),
        #'trials_2'  : np.array([row[0][1][0][0] for row in mat2['trials']]),
        #'times'     : np.array([datetime.strptime(row[0][2][0], "%H:%M:%S.%f") for row in mat2['trials']]),
        #'???_1'     : np.array([row[0][3] for row in mat2['trials']]),
        #'???_2'     : np.array([row[0][4][0][0] for row in mat2['trials']]),
        #'TEXTURE_NAMES'  : list(set([row[0][5][0] for row in mat2['trials']])),
        #'TEXTURE_TYPES'  : np.array([row[0][5][0] for row in mat2['trials']]),
        #'???_4'     : np.array([row[0][6][0] for row in mat2['trials']]),
        #'???_5'     : np.array([row[0][7][0][0] for row in mat2['trials']]),
        #'???_6'     : np.array([row[0][8][0] for row in mat2['trials']]),
        #'???_7'     : np.array([row[0][9][0][0] for row in mat2['trials']]),
        #'RESPONSE_NAMES' : list(set([row[0][10][0] for row in mat2['trials']])),
        #'RESPONSE_TYPES' : np.array([row[0][10][0] for row in mat2['trials']]),
        #'???_8'     : np.array([row[0][11][0][0] for row in mat2['trials']]),
        #'???_9'     : np.array([row[0][12][0][0] for row in mat2['trials']])
    #}

    ## Enumerate textures and responses
    #d_behaviour['TEXTURE_TYPES'] = np.array([d_behaviour['TEXTURE_NAMES'].index(texname) for texname in d_behaviour['TEXTURE_TYPES']])
    #d_behaviour['RESPONSE_TYPES'] = np.array([d_behaviour['RESPONSE_NAMES'].index(texname) for texname in d_behaviour['RESPONSE_TYPES']])
    #mat2['trials'] = d_behaviour
    mat2['trials'] = {}
    
    return mat1['data'], mat2


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
