import numpy as np
import scipy.stats
from scipy import interpolate

# Spearmann rank matrix
def spMat(data):
    N = data.shape[0]
    spMat = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            spMat[i][j] = scipy.stats.spearmanr(data[i], data[j])[0]
    spMat += spMat.transpose()
    spMat += np.diag([1]*N)
    return spMat

# DEFINE BINNED DATA AS AVERAGE VALUE OF POINS IN THE BIN
def downsample(x1, y1, DX2, INTERP_SCALE = 10):
    N1 = len(x1)
    X_RANGE = (np.min(x1), np.max(x1))
    X_LEN = X_RANGE[1] - X_RANGE[0]

    # Number of bins
    N2 = int(X_LEN / DX2)

    # Centers of bins
    x2 = np.linspace(X_RANGE[0], X_RANGE[1] - DX2, N2) + DX2 / 2

    # 1. Interpolate y1
    f_interp = interpolate.interp1d(x1, y1, kind='linear')

    # 2. Sample function using a lot of points
    x_interp = np.linspace(X_RANGE[0], X_RANGE[1], N1*INTERP_SCALE)
    y_interp = f_interp(x_interp)

    # 3. Add up using slices
    N_POINT_PER_BIN = int(len(x_interp) / N2)
    y2 = [np.mean(y_interp[N_POINT_PER_BIN*i: N_POINT_PER_BIN*(i+1)]) for i in range(N2)]
    
    #print(N2, N_POINT_PER_BIN, len(x_interp))

    return x2, y2