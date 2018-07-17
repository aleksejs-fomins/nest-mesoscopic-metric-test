import numpy as np

def approxDelayConv(data, TAU, DT):
    rez = np.zeros(len(data)+1)
    for i in range(1, len(data)+1):
        rez[i] = data[i-1] + rez[i-1] * (1 - DT / TAU)

    return rez[1:]

def samplingRangeScale(x, delta, tau):
    return np.multiply(x < delta, 1.0) + np.multiply(x >= delta, np.exp(-(x-delta)/tau))


# Take discretized spiking signal, produce macroscopic CA indicator response
def spike2ca(spikeTimes, neuronIdxs, DT, TAU_CA_IND, MIN_TIME, MAX_TIME, samplingRange):

    # Get properties of the distribution
#     MIN_TIME = np.min(spikeTimes)
#     MAX_TIME = np.max(spikeTimes)
    MIN_NEUR_ID = int(np.min(neuronIdxs))
    MAX_NEUR_ID = int(np.max(neuronIdxs))
    N_NEURON = MAX_NEUR_ID - MIN_NEUR_ID + 1

    # Simulate variability of Ca indicator response strength,
    # and visibility of individual neurons via random factor
    neuronVariability = samplingRangeScale(np.random.uniform(0, 1, N_NEURON), samplingRange, 0.1)

    # Take average spike trace, considering variability
    times_discr = np.arange(MIN_TIME, MAX_TIME + DT, DT)
    N_TIME_STEP = len(times_discr)
    signalCaAvg = np.zeros(N_TIME_STEP)

    for t, n in zip(spikeTimes, neuronIdxs):
        signalCaAvg[int((t - MIN_TIME) / DT)] += neuronVariability[int(n) - MIN_NEUR_ID]

    # Convolve time-signal with indicator response
    signalCaAvg = approxDelayConv(signalCaAvg, TAU_CA_IND, DT)

    return times_discr, signalCaAvg

    ###########################################################################
    # Code below is more detailed, for the case of variability in neural response. Much slower though
    ###########################################################################

    # # # Construct indicator response function
    # # indicatorResponse_t = np.arange(0, 6 * TAU_CA_IND, DT)
    # # indicatorResponse_x = np.exp(-indicatorResponse_t / TAU_CA_IND)
    #
    # # Separate data into individual traces
    # times_discr = np.arange(MIN_TIME, MAX_TIME + DT, DT)
    # N_TIME_STEP = len(times_discr)
    #
    # # print(MIN_TIME, MAX_TIME, DT, N_TIME_STEP)
    #
    # signalCaAvg = np.zeros(N_TIME_STEP)
    #
    #
    # for i in range(MIN_NEUR_ID, MAX_NEUR_ID + 1):
    #     # Extract spike times of this neuron
    #     spikeTimesThis = spikeTimes[neuronIdxs == i]
    #     spikeTimeIdxs = ((spikeTimesThis - MIN_TIME) / DT).astype(int)
    #
    #     # Convert them into time-continuous signal
    #     signalDiscr = np.zeros(N_TIME_STEP)
    #     signalDiscr[spikeTimeIdxs] = 1
    #
    #     # Convolve time-signal with indicator response
    #     # signalCa = np.convolve(indicatorResponse_x, signalDiscr)[:N_TIME_STEP]
    #     signalCa = approxDelayConv(signalDiscr, TAU_CA_IND, DT)
    #
    #     # Average over all cells in the area
    #     # Simulate variability of Ca indicator response, and visibility of individual neurons via random factor
    #     signalCaAvg += signalCa / N_NEURON * np.random.uniform(0, 1)
    #
    # return times_discr, signalCaAvg