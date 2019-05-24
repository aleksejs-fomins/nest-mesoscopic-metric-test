import os

# Merge 2 dictionaries, given that values of both are lists
def merge_dicts(d_lst):
    d_rez = d_lst[0]
    for i in range(1, len(d_lst)):
        d_rez = {k1 : v1 + d_lst[i][k1] for k1, v1 in d_rez.items()}
    return d_rez

def get_subfolders(folderpath):
    return [_dir for _dir in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, _dir))]