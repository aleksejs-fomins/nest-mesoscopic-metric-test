import os, sys
from datetime import datetime
import numpy as np
import scipy.io

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from matlab.matlab_lib import loadmat, matstruct2dict
from matlab.aux_functions import merge_dicts, get_subfolders

# Read data and behaviour matlab files given containing folder
def read_neuro_perf(folderpath):
    # Read MAT file from command line
    print("Reading Yaro data from", folderpath)
    datafilename = os.path.join(folderpath, "data.mat")
    behaviorfilename = os.path.join(folderpath, "behaviorvar.mat")

    data = loadmat(datafilename)['data']
    behavior = loadmat(behaviorfilename)
    
    # Get rid of useless fields in behaviour
    behavior = {k : v for k, v in behavior.items() if k[0] != '_'}
    
    # Convert trials structure to a dictionary
    behavior['trials'] = merge_dicts([matstruct2dict(obj) for obj in behavior['trials']])
    
    # d_trials = matstruct2dict(behavior['trials'][0])
    # for i in range(1, len(behavior['trials'])):
    #     d_trials = merge_dict(d_trials, matstruct2dict(behavior['trials'][i]))
    # behavior['trials'] = d_trials
    
    return data, behavior

# Read multiple neuro and performance files from a root folder
def read_neuro_perf_multi(rootpath):
    
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
            data, behaviour = read_neuro_perf(daypath)
            micedict[mouse][day] = {
                'data' : data,
                'behaviour' : behaviour
            }
            
    return micedict

def read_lick(folderpath):        
    rez = {}
    
    ################################
    # Process Reaction times file
    ################################
    rt_file = os.path.join(folderpath, "RT_264.mat")
    rt = loadmat(rt_file)
    
    rez['reaction_time'] = 3.0 + rt['reaction_time']
    
    ################################
    # Process lick_traces file
    ################################
    def lick_filter(data, bot_th, top_th):
        data[np.isnan(data)] = 0
        return np.logical_or(data <= bot_th, data >= top_th).astype(int)
    
    lick_traces_file = os.path.join(folderpath, "lick_traces.mat")
    lick_traces = loadmat(lick_traces_file)
    
    nTimesLick = len(lick_traces['licks_go'])
    freqLick = 100 # Hz
    rez['tLicks'] = np.linspace(0, (nTimesLick-1) / freqLick, nTimesLick)
    
    # Top threshold is wrong sometimes. Yaro said to use exact one
    thBot, thTop = lick_traces['bot_thresh'], 2.64
    
    for k in ['licks_go', 'licks_nogo', 'licks_miss', 'licks_FA', 'licks_early']:
        rez[k] = lick_filter(lick_traces[k], thBot, thTop)
        
    ################################
    # Process trials file
    ################################
    TIMESCALE_TRACES = 0.001 # ms
    trials_file = os.path.join(folderpath, os.path.basename(folderpath)+".mat")
    lick_trials = loadmat(trials_file)

    # NOTE: lick_trials['licks']['lick_vector'] is just a repeat from above lick_traces file
#     lick_trials['licks'] = merge_dicts([matstruct2dict(obj) for obj in lick_trials['licks']])
    lick_trials['trials'] = merge_dicts([matstruct2dict(obj) for obj in lick_trials['trials']])
    rez['reward_time'] = np.array(lick_trials['trials']['reward_time'], dtype=float) * TIMESCALE_TRACES
    rez['puff'] = [np.array(puff, dtype=float)*TIMESCALE_TRACES for puff in lick_trials['trials']['puff']]
        
    return rez


def read_paw(filepath):
    paw_trials = loadmat(filepath)['trials']
    
    nTrialsPaw, nTimePaw = paw_trials.shape
    if nTimePaw == 64:
        freqPaw = 7
    elif nTimePaw > 250:
        freqPaw = 30
    else:
        raise ValueError("Unexpected number of paw timesteps", nTimePaw)

    return {
        'tPaw' : np.linspace(0, (nTimePaw-1) / freqPaw, nTimePaw),
        'trialsPaw' : paw_trials,
        'freqPaw' : freqPaw
    }

def read_whisk(folderpath):
    # whiskAngle = loadmat(os.path.join(folderpath, 'whiskAngle.mat'))['whiskAngle']
    # nTimesWhisk, nTrialsWhisk = whiskAngle.shape
    # if nTimesWhisk <= 400:
    #     freqWhisk = 40
    # elif nTimesWhisk >= 1600:
    #     freqWhisk = 200
    # else:
    #     raise ValueError("Unexpected number of paw timesteps", nTimesWhisk)
        
    with open(os.path.join(folderpath, os.path.basename(folderpath)+'.txt')) as fLog:
        firstTouch = np.array([line.split('\t')[1] for line in fLog.readlines()[1:]], dtype=float)
        
    # return {
    #     'tWhisk' : np.linspace(0, (nTimesWhisk-1) / freqWhisk, nTimesWhisk),
    #     'whiskAngle' : whiskAngle,
    #     #'whiskAbsVelocity' : np.vstack((np.abs(whiskAngle[1:] - whiskAngle[:-1])*freqWhisk, np.zeros(nTrialsWhisk))),
    #     'firstTouch' : firstTouch
    # }
    return {'firstTouch' : firstTouch}

    
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