import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.signal
import nest
import h5py
from spike2ca import spike2ca


class BrunnelMesoscopic():
    def __init__(self, param):
        self.param = param

        # POPULATION PROPERTIES
        self.N_REGION = param['N_REGION']
        self.N_POPULAITONS = 2 * param['N_REGION']
        N_E = int(0.8 * param['NETWORK_SIZE_FACTOR'])
        N_I = int(0.2 * param['NETWORK_SIZE_FACTOR'])
        self.N_NEURON_PP = [N_E, N_I] * param['N_REGION']

        # Synapse properties
        W_EXC_MAX = param['WSCALE_INTERN']
        W_INH_MAX = -4 * param['WSCALE_INTERN']

        # Connectivity
        # Inter-layer connectivity graph
        self.N_CONN_EXTERN = len(param['MESOSCOPIC_CONN'])
        self.CONN_MESOSCOPIC_POPULATIONS = [(2*i, 2*j) for i,j in param['MESOSCOPIC_CONN']]
        self.CONN_GRAPH_POPULATIONS = []

        # Construct connectivity within each layer
        for i in range(param['N_REGION']):
            self.CONN_GRAPH_POPULATIONS += [
                (2*i,   2*i,   {'WMAX' : W_EXC_MAX, 'PCONN' : param['PCONN_INTERN'][0], 'DELAY' : param['SYN_TAU_INTERN']}),
                (2*i,   2*i+1, {'WMAX' : W_EXC_MAX, 'PCONN' : param['PCONN_INTERN'][1], 'DELAY' : param['SYN_TAU_INTERN']}),
                (2*i+1, 2*i,   {'WMAX' : W_INH_MAX, 'PCONN' : param['PCONN_INTERN'][2], 'DELAY' : param['SYN_TAU_INTERN']}),
                (2*i+1, 2*i+1, {'WMAX' : W_INH_MAX, 'PCONN' : param['PCONN_INTERN'][3], 'DELAY' : param['SYN_TAU_INTERN']})
            ]

        # Construct inter-layer connectivity
        for iReg in range(self.N_CONN_EXTERN):
            i,j = param['MESOSCOPIC_CONN'][iReg]
            self.CONN_GRAPH_POPULATIONS += [(2*i, 2*j,   {'WMAX' : W_EXC_MAX, 'PCONN' : param['PCONN_EXTERN'][iReg], 'DELAY' : param['SYN_TAU_EXTERN']})]


    # Plot connectivity graph
    def plotConnectivity(self):
        def polar(r, phi):
            return r * np.array([np.cos(phi), np.sin(phi)])

        def makeArrow(pos, col, ax):
            kw = dict(arrowstyle="Simple,tail_width=1,head_width=8,head_length=8", color=col)
            posPrim = [0.95*pos[0] + 0.05*pos[1], 0.05*pos[0] + 0.95*pos[1]]
            arrow123 = patches.FancyArrowPatch(posPrim[0], posPrim[1], connectionstyle="arc3,rad=.5", **kw)
            ax.add_patch(arrow123)


        graph_r = 10
        graph_rc = [50, 80]
        graph_color_type = ['blue', 'red']

        ctype = [i % 2 for i in range(self.N_POPULAITONS)]  # Cell types
        graph_phi = [2 * np.pi * (i // 2) / self.N_REGION for i in range(self.N_POPULAITONS)]
        graph_coord = [polar(graph_rc[ctype[i]], graph_phi[i]) for i in range(self.N_POPULAITONS)]
        graph_color = [graph_color_type[ctype[i]] for i in range(self.N_POPULAITONS)]

        # Nodes
        fig, ax = plt.subplots(figsize = (10, 10))
        for i in range(self.N_POPULAITONS):
            circ = plt.Circle(graph_coord[i], graph_r, color=graph_color[i])
            ax.add_patch(circ)

        # Connections
        for i, j, param in self.CONN_GRAPH_POPULATIONS:
            color = 'k' if np.abs(i - j) == 1 else 'y'
            makeArrow([graph_coord[i], graph_coord[j]], color, ax)

        plt.axis('off')
        ax.set_title('Population Connectivity')
        ax.set_xlim(-(graph_rc[1]+graph_r), (graph_rc[1]+graph_r))
        ax.set_ylim(-(graph_rc[1]+graph_r), (graph_rc[1]+graph_r))


    # Construct the graph and run
    def run(self, tstop, simParam):
        self.dt = simParam['resolution']
        self.tstop = tstop

        # reset kernel and set dt
        nest.ResetKernel()
        nest.SetKernelStatus(simParam)

        # create nodes
        self.neuron_nodes = [nest.Create('iaf_psc_alpha', n) for n in self.N_NEURON_PP]
        self.neuron_spike_detectors = [nest.Create('spike_detector') for i in range(self.N_POPULAITONS)]
        self.noise_generator = nest.Create('poisson_generator', params={'rate': self.param['NOISE_RATE']})

        # Connect input to 0th node
        if self.param['STIM_MAG'] > 0.0:
            ac_generator = nest.Create('ac_generator', params={'amplitude': self.param['STIM_MAG'], 'frequency': self.param['STIM_FREQ']})
            nest.Connect(ac_generator, self.neuron_nodes[0], conn_spec='all_to_all')

        # create connections
        for i in range(self.N_POPULAITONS):
            # Connect detectors to populations
            nest.Connect(self.neuron_nodes[i], self.neuron_spike_detectors[i], conn_spec='all_to_all')

            # Connect noise to populations
            nest.Connect(self.noise_generator, self.neuron_nodes[i],
                         conn_spec='all_to_all',
                         syn_spec={'weight': self.param['W_NOISE_MAX'], 'delay': self.param['SYN_TAU_NOISE']})

        for i, j, param in self.CONN_GRAPH_POPULATIONS:
            # Connect populations to each other
            nest.Connect(self.neuron_nodes[i], self.neuron_nodes[j],
                         conn_spec={'rule': 'pairwise_bernoulli', 'p': param['PCONN']},
                         syn_spec={'model': 'static_synapse', 'weight': param['WMAX'], 'delay': param['DELAY']})

        # simulate
        nest.Simulate(tstop)


    # Save connectivity to HDF5
    def saveConnectivity(self, basename):
        self.basename = basename
        f = h5py.File(basename + ".hdf5", "w")
        f['N_POPULAITONS'] = self.N_POPULAITONS
        f['REGION_START'] = [nest.GetStatus(self.neuron_nodes[i], 'global_id')[0] for i in range(self.N_POPULAITONS)]
        f['REGION_END']   = [nest.GetStatus(self.neuron_nodes[i], 'global_id')[-1] for i in range(self.N_POPULAITONS)]
        f['POPULATION_CONN'] = [(2 * i, 2 * j) for i, j in self.param['MESOSCOPIC_CONN']]
        f['dt'] = self.dt
        f['T_RANGE'] = (0, self.tstop)
        f.close()


    # Save traces to HDF5
    def saveTraces(self):
        f = h5py.File(self.basename + ".hdf5", "r+")
        for i in range(self.N_POPULAITONS):
            f['SPIKE_TIMES_' + str(i)] = nest.GetStatus(self.neuron_spike_detectors[i])[0]['events']['times']
            f['SPIKE_SENDERS_' + str(i)] = nest.GetStatus(self.neuron_spike_detectors[i])[0]['events']['senders']
        f.close()


    # COMPUTE ENERGY TRANSFER and save to HDF5
    # TODO: IMPLEMENT TIME_DELAY INTO ENERGY TRANSFER
    def saveEnergyTransfer(self, withPlot = False):

        #######################################
        # 1. Extract inter-layer connectivity
        #######################################

        # Learn to map from neuron index to population index
        # Convert to zero-starting index
        maxNeuronIdx = [nest.GetStatus(self.neuron_nodes[i], 'global_id')[-1]-1 for i in range(self.N_POPULAITONS)]
        N_NEURON_TOT = maxNeuronIdx[-1]  # There are also spike detectors and such that come with larger indices

        def getPopIdx(neuronIdx):
            i = 0
            while maxNeuronIdx[i] <= neuronIdx:
                i += 1
            return i

        # Find out how many times each neuron maps to another region
        connMap = np.zeros((N_NEURON_TOT, self.N_CONN_EXTERN))
        for arr in nest.GetConnections():
            src = arr[0] - 1
            trg = arr[1] - 1

            if (src < N_NEURON_TOT) and (trg < N_NEURON_TOT):
                linkIdx = (getPopIdx(src), getPopIdx(trg))

                if linkIdx in self.CONN_MESOSCOPIC_POPULATIONS:
                    connMap[src][self.CONN_MESOSCOPIC_POPULATIONS.index(linkIdx)] += self.param['WSCALE_INTERN']

        ######################################################
        # 2. Convert spikes into cross-layer energy transfer
        ######################################################

        MIN_TIME = 0
        MAX_TIME = self.tstop
        times_discr = np.arange(MIN_TIME, MAX_TIME + self.dt, self.dt)
        micro_energy = np.zeros((self.N_CONN_EXTERN, len(times_discr)))

        for iPop in range(self.N_POPULAITONS):
            # Only consider excitatory population data
            if (iPop % 2 == 0):
                tspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['times']
                nspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['senders']

                for t, n in zip(tspk, nspk):
                    timeIdx = int((t - MIN_TIME) / self.dt)
                    for iLink in range(self.N_CONN_EXTERN):
                        micro_energy[iLink][timeIdx] += connMap[n - 1][iLink]

        # Write to file
        f = h5py.File(self.basename + ".hdf5", "r+")
        f['ENERGY_TRANSFER_T'] = times_discr
        f['ENERGY_TRANSFER_X'] = micro_energy
        f.close()

        # Plot if requested
        if withPlot:
            plt.figure()
            for i in range(len(micro_energy)):
                plt.plot(times_discr, micro_energy[i])


    def saveGCampTraces(self, param, withPlot = False):
        # Coarsen the input to account for finite camera framerate
        # Delete last time point because it is incomplete due to coarsening
        MIN_TIME = 0
        MAX_TIME = self.tstop
        BINSIZE = 1000 / param['FPS']
        tCoarse = np.linspace(MIN_TIME, MAX_TIME, int((MAX_TIME-MIN_TIME)/BINSIZE)+1)
        xCoarseLst = np.zeros((self.N_POPULAITONS, len(tCoarse)-1))

        # Write to file
        f = h5py.File(self.basename + ".hdf5", "r+")
        f['GCAMP_COARSE_T'] = tCoarse[:-1]


        # Plot if requested
        if withPlot:
            plt.figure()

        for iPop in range(self.N_POPULAITONS):
            tspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['times']
            nspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['senders']

            dataFine = spike2ca(tspk, nspk, self.dt, param['TAU_CA'], MIN_TIME, MAX_TIME, param['GEOM_RANGE'])

            xCoarse = np.zeros(len(tCoarse))
            for t, x in zip(dataFine[0], dataFine[1]):
                xCoarse[int((t - MIN_TIME) / BINSIZE)] += x / BINSIZE

            # Write to file
            xCoarseLst[iPop] = xCoarse[:-1]

            # Plot if requested
            if withPlot:
                plt.plot(tCoarse[:-1], xCoarse[:-1])

        # Close HDF5
        f['GCAMP_COARSE_X'] = xCoarseLst
        f.close()
