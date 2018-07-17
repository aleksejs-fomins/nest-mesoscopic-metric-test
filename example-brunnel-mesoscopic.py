import time
import matplotlib.pyplot as plt
import brunelmesoscopic

modelParam = {
    'N_REGION'                  : 5,
    'MESOSCOPIC_CONN'           : [(0, 1), (1, 2), (3, 1), (4, 2)],

    'NETWORK_SIZE_FACTOR'       : 1000,
    'PCONN_INTERN'              : [0.2, 0.2, 0.2, 0.2],       # EXC-EXC, EXC-INH, INH-EXC, INH-INH
    'PCONN_EXTERN'              : [0.05, 0.05, 0.05, 0.05],   # One for each region

    'WSCALE_INTERN'             : 2.0,
    'WSCALE_EXTERN'             : 2.0,
    'SYN_TAU_INTERN'            : 1.5,
    'SYN_TAU_EXTERN'            : 200.0,

    'NOISE_RATE'                : 6000.0,
    'W_NOISE_MAX'               : 10.0,
    'SYN_TAU_NOISE'             : 0.1,

    'STIM_MAG'                  : 0.0,
    'STIM_FREQ'                 : 50.0
}

# Create the simulation
BM1 = brunelmesoscopic.BrunnelMesoscopic(modelParam)

# Plot Connectivity
BM1.plotConnectivity()
plt.show()

# Run the simulation
tstop = 8000.    # simulation duration 8sec
simulationParam = {
    'resolution': 0.1,          # time discretization, ms
    'local_num_threads': 4,     # number of parallel threads to use
}

start = time.time()
BM1.run(tstop, simulationParam)
print("Time taken to simulate:", time.time() - start)

# Write results to file
start = time.time()
BM1.saveConnectivity("datafile")
# BM1.saveTraces()
BM1.saveEnergyTransfer()
print("Time taken to write results to file:", time.time() - start)

# Compute and write GCamp traces to file
gcampParam = {
    'TAU_CA'        : 400,  # (ms) Delay of CA indicator
    'GEOM_RANGE'    : 0.5,  # Ratio of neurons that will be 100% visible to the optical detector
    'FPS'           : 50.0
}

start = time.time()
BM1.saveGCampTraces(gcampParam, withPlot = False)
print("Time taken to compute GCamp traces:", time.time() - start)