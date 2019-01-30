import numpy as np
from scipy import interpolate


def gaussian(mu, s2):
    return np.exp(- mu**2 / (2 * s2) )


# Compute discretized exponential decay convolution
def approxDelayConv(data, TAU, DT):
    N = data.shape[0]
    alpha = DT / TAU
    beta = 1-alpha
    
    rez = np.zeros(N+1)
    for i in range(1, N+1):
        rez[i] = data[i-1]*alpha + rez[i-1]*beta

    return rez[1:]

# # Imitate geometric sampling, by selecting some neurons 100% and the rest exponentially dropping
# def samplingRangeScale(x, delta, tau):
#     return np.multiply(x < delta, 1.0) + np.multiply(x >= delta, np.exp(-(x-delta)/tau))

# Downsample uniformly-sampled data by bin-averaging
def downsample(x1, y1, N2, method="window", ker_sig2=None):
    # Find number of points and spacings for original and downsampled datasets
    N1 = len(x1)
    DX1 = x1[1] - x1[0]
    X_LEN_EXT = DX1 * N1
    DX2 = X_LEN_EXT / N2
    
    # Find optimal position of downsampled coordinates
    x2_start = x1[0]  + 0.5*(DX2 - DX1)
    x2_end   = x1[-1] - 0.5*(DX2 - DX1)
    x2 = np.linspace(x2_start, x2_end, N2)
    
    if ker_sig2 == None:
        ker_sig2 = (DX2/2)**2
    
    # Find downsampled values as averages over corresponding windows
    y2 = np.zeros(N2)
    for i2 in range(N2):
        if method == "window":
            # Find time-window to average
            w_l = x2[i2] - 0.5 * DX2
            w_r = x2[i2] + 0.5 * DX2

            # Find points of original dataset to average
            i1_l = np.max([int(np.ceil((w_l - x1[0]) / DX1)), 0])
            i1_r = np.min([int(np.floor((w_r - x1[0]) / DX1)), N1])

            # Compute downsampled values by averaging
            y2[i2] = np.mean(y1[i1_l:i1_r])
        else:
            # Each downsampled val is average of all original val weighted by proximity kernel
            w_ker = gaussian(x2[i2] - x1, ker_sig2)
            w_ker /= np.sum(w_ker)
            y2[i2] = w_ker.dot(y1)
        
    return x2, y2


# # DEFINE BINNED DATA AS AVERAGE VALUE OF POINS IN THE BIN
# def downsample_interp(x1, y1, DX2, INTERP_SCALE = 10):
#     N1 = len(x1)
#     X_RANGE = (np.min(x1), np.max(x1))
#     X_LEN = X_RANGE[1] - X_RANGE[0]

#     # Number of bins
#     N2 = int(X_LEN / DX2)

#     # Centers of bins
#     x2 = np.linspace(X_RANGE[0], X_RANGE[1] - DX2, N2) + DX2 / 2

#     # 1. Interpolate y1
#     f_interp = interpolate.interp1d(x1, y1, kind='linear')

#     # 2. Sample function using a lot of points
#     x_interp = np.linspace(X_RANGE[0], X_RANGE[1], N1*INTERP_SCALE)
#     y_interp = f_interp(x_interp)

#     # 3. Add up using slices
#     N_POINT_PER_BIN = int(len(x_interp) / N2)
#     y2 = [np.mean(y_interp[N_POINT_PER_BIN*i: N_POINT_PER_BIN*(i+1)]) for i in range(N2)]

#     return x2, y2