import numpy as np
import os, sys

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from os_lib import getfiles_walk

def parseFoldersMulti(root_path_data, root_path_paw, root_path_lick, root_path_whisk):
    ##################################
    # Get channel labels for all mice
    ##################################
    labels_paths = getfiles_walk(root_path_data, ['channel_labels.mat'])

    micePathArr = []
    for path, name in labels_paths:
        mousename = os.path.basename(path)
        micePathArr += [[mousename, 'channel_labels', os.path.join(path, name), mousename]]

    ##################################
    # Get data folders for all days
    ##################################
    data_paths = getfiles_walk(root_path_data, ["data.mat"])
    for path, name in data_paths:
        mouseday = os.path.basename(path)
        mousename = os.path.basename(os.path.dirname(path))
        micePathArr += [[mousename, 'data_path', path, mouseday]]

    ##################################
    # Get paw folders for all mice
    ##################################
    paw_paths = getfiles_walk(root_path_paw, ["deltaI_paw.mat"])
    for path, name in paw_paths:
        mouseday = os.path.basename(path)
        mousename = os.path.basename(os.path.dirname(path))
        micePathArr += [[mousename, 'paw_path', path, mouseday]]

    ##################################
    # Get lick folders for all mice
    ##################################
    lick_paths = getfiles_walk(root_path_lick, ["lick_traces.mat"])
    for path, name in lick_paths:
        mouseday = os.path.basename(path)
        mousename = os.path.basename(os.path.dirname(path))
        micePathArr += [[mousename, 'lick_path', path, mouseday]]

    ##################################
    # Get whisk folders for all mice
    ##################################
    whisk_paths = getfiles_walk(root_path_whisk, ["whiskAngle.mat"])
    for path, name in whisk_paths:
        mouseday = os.path.basename(path)
        mousename = os.path.basename(os.path.dirname(path))
        micePathArr += [[mousename, 'whisk_path', path, mouseday]]


    ##################################
    # Print results
    ##################################
    micePathArr = np.array(micePathArr)
    mice          = list(set(micePathArr[:, 0]))
    data_subArr   = micePathArr[micePathArr[:, 1]=='data_path']
    paw_subArr    = micePathArr[micePathArr[:, 1]=='paw_path']
    lick_subArr   = micePathArr[micePathArr[:, 1]=='lick_path']
    whisk_subArr  = micePathArr[micePathArr[:, 1]=='whisk_path']

    sumByMouse = lambda subarr : [len(subarr[subarr[:, 0] == mousename]) for mousename in mice]

    summary = {
        "Data"  : sumByMouse(data_subArr),
        "Paw"   : sumByMouse(paw_subArr),
        "Lick"  : sumByMouse(lick_subArr),
        "Whisk" : sumByMouse(whisk_subArr)
    }
    
    return micePathArr, summary