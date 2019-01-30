import numpy as np
import pandas as pd

# Convert results structure into Pandas dataframe for better usability
def idtxlResults2Pandas(results, N_NODE):
    
    cols = ['src', 'trg', 'te', 'lag', 'p']
    df = pd.DataFrame([], columns=cols)
    
    for i in range(N_NODE):
        rezThis = results.get_single_target(i, fdr=False)
        
        # Convert numpy arrays to lists to concatenate
        # Make sure that if None returned, replace with empty list
        # none2lst = lambda l: list(l) if l is not None else []
        
        # If any connections were found, get their data  at all was found
        if rezThis['selected_sources_te'] is not None:
            te  = rezThis['selected_sources_te']
            p   = rezThis['selected_sources_pval']
            lag = [val[1] for val in rezThis['selected_vars_sources']]
            src = [val[0] for val in rezThis['selected_vars_sources']]
            trg = [i] * len(te)

            df = df.append(pd.DataFrame(list(zip(src, trg, te, lag, p)), columns=cols), ignore_index=True)
    
#     df = pd.DataFrame.from_dict(out)
    df = df.sort_values(by=['src', 'trg'])
    
    return df


# Convert results structure into set of matrices for better usability
def idtxlResults2matrix(results, N_NODE, method='TE'):
    if 'TE' in method:
        metric_name = 'selected_sources_te'
    elif 'MI' in method:
        metric_name = 'selected_sources_mi'
    else:
        raise ValueError('Unexpected method', method)
    
    
    te_mat = np.zeros((N_NODE, N_NODE)) + np.nan
    lag_mat = np.zeros((N_NODE, N_NODE)) + np.nan
    p_mat = np.zeros((N_NODE, N_NODE)) + np.nan
    
    for iTrg in range(N_NODE):
        rezThis = results.get_single_target(iTrg, fdr=False)
        
        # If any connections were found, get their data  at all was found
        if rezThis[metric_name] is not None:
            rezThisZip = zip(
                rezThis[metric_name],
                rezThis['selected_sources_pval'],
                [val[1] for val in rezThis['selected_vars_sources']],
                [val[0] for val in rezThis['selected_vars_sources']]
            )
            
            for te, p, lag, iSrc in rezThisZip:
                te_mat[iSrc][iTrg] = te
                lag_mat[iSrc][iTrg] = lag
                p_mat[iSrc][iTrg] = p
    
    return te_mat, lag_mat, p_mat
