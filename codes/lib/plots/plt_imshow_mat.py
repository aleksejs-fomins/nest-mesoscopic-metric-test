from matplotlib import colors
import matplotlib.pyplot as plt

def plotImshowMat(mat_lst, title_lst, title, shape, lims=None, draw=False):
    
    # Create plot matrix
    nRows, nCols = shape
    fig, ax = plt.subplots(nrows=nRows, ncols=nCols)
    fig.suptitle(title)
    
    # Convert plot indices to 1D index
    ax1D = ax.flatten()
    n1D = nRows*nCols
    
    # Plot data
    for i in range(n1D):
        pl = ax1D[i].imshow(mat_lst[i], cmap='jet')
        ax1D[i].set_title(title_lst[i])
        fig.colorbar(pl, ax=ax1D[i])
        
        if lims is not None:
            norm = colors.Normalize(vmin=lims[i][0], vmax=lims[i][1])
            pl.set_norm(norm)

    if draw:
        plt.draw()
    else:
        plt.show()
