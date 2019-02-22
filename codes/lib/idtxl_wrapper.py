#import sys
#import contextlib
import numpy as np
import pandas as pd
import pathos

# IDTxl libraries
from idtxl.bivariate_mi import BivariateMI
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network

def idtxlParallelCPU(data, settings):
    # Get number of processes
    idxProcesses = settings['dim_order'].index("p")
    NProcesses = data.shape[idxProcesses]
    
    # Convert data to ITDxl format
    dataIDTxl = Data(data, dim_order=settings['dim_order'])
    
    # Initialise analysis object
    if   settings['method'] == "BivariateMI":     analysis_class = BivariateMI()
    elif settings['method'] == "MultivariateMI":  analysis_class = MultivariateMI()
    elif settings['method'] == "BivariateTE":     analysis_class = BivariateTE()
    elif settings['method'] == "MultivariateTE":  analysis_class = MultivariateTE()
    else:
        raise ValueError("Unexpected method", settings['method'])

    # Initialize multiprocessing pool
    NCore = pathos.multiprocessing.cpu_count() - 1
    pool = pathos.multiprocessing.ProcessingPool(NCore)
    #pool = multiprocessing.Pool(NCore)
    
    # Run analysis
    # Temporarily move all output to a log-file, as IDTxl likes output
    
    #with contextlib.redirect_stdout(open('log_out.txt', 'w')):
    #    with contextlib.redirect_stderr(open('log_err.txt', 'w')):
    targetLst = list(range(NProcesses))
    parallelTask = lambda trg: analysis_class.analyse_single_target(settings=settings, data=dataIDTxl, target=trg)
    rez = pool.map(parallelTask, targetLst)
    return rez


# Convert results structure into set of matrices for better usability
def idtxlResultsParse(results, N_NODE, method='TE', storage='matrix'):
    # Determine metric name to be extracted
    if 'TE' in method:
        metric_name = 'selected_sources_te'
    elif 'MI' in method:
        metric_name = 'selected_sources_mi'
    else:
        raise ValueError('Unexpected method', method)
    
    # Initialize target storage class
    if storage=='pandas':
        cols = ['src', 'trg', 'te', 'lag', 'p']
        df = pd.DataFrame([], columns=cols)
    else:    
        te_mat = np.zeros((N_NODE, N_NODE)) + np.nan
        lag_mat = np.zeros((N_NODE, N_NODE)) + np.nan
        p_mat = np.zeros((N_NODE, N_NODE)) + np.nan
    
    # Parse data
    for iTrg in range(N_NODE):
        if isinstance(results, list):
            rezThis = results[iTrg].get_single_target(iTrg, fdr=False)
        else:
            rezThis = results.get_single_target(iTrg, fdr=False)
        
        # If any connections were found, get their data  at all was found
        if rezThis[metric_name] is not None:
            te_lst  = rezThis[metric_name]
            p_lst   = rezThis['selected_sources_pval']
            lag_lst = [val[1] for val in rezThis['selected_vars_sources']]
            src_lst = [val[0] for val in rezThis['selected_vars_sources']]
            trg_lst = [iTrg] * len(te_lst)
            rezThisZip = zip(src_lst, trg_lst, te_lst, lag_lst, p_lst)
            
            if storage=='pandas':
                df = df.append(pd.DataFrame(list(rezThisZip), columns=cols), ignore_index=True)
            else:            
                for iSrc, iTrg, te, lag, p in rezThisZip:
                    te_mat[iSrc][iTrg] = te
                    lag_mat[iSrc][iTrg] = lag
                    p_mat[iSrc][iTrg] = p
    if storage=='pandas':
        #df = pd.DataFrame.from_dict(out)
        df = df.sort_values(by=['src', 'trg'])
        return df        
    else:
        return te_mat, lag_mat, p_mat