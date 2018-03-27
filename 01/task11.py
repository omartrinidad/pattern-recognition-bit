import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from auxiliar import *


@save_figure()
def plotData2D(X, path = ""):

    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')

    # plot the data
    axs.plot(X[0,:], X[1,:], 'o', label='data', alpha=0.333, c="#2222ee")

    # set x and y limits of the plotting area
    xmin, xmax = X[0,:].min()-5, X[0,:].max()+5
    ymin, ymax = X[1,:].min()-5, X[1,:].max()+5

    axs.set_facecolor("#eeeeff")
    axs.set_xlim(xmin, xmax)
    axs.set_ylim(ymin, ymax)

    # set properties of the legend of the plot
    # leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    # leg.get_frame().set_alpha(0.5)

    plt.xlabel("Height in centimeters")
    plt.ylabel("Weight in kilograms")

    # either show figure on screen or write it to disk
    return plt


if __name__ == "__main__":

    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('data/whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data])


    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('data/whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)

    X = data[:,0:2].astype(np.float)

    data = data[np.all(X>=0, 1)]
    X = data[:,0:2].astype(np.float)
    y = data[:,2]
    # read gender data into 1D array (i.e. into a vector)


    # let's transpose the data matrix
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotData2D(X, path="latex/outliers1.tex")

    # next, let's plot height vs. weight
    # first, copy information rows of X into 1D arrays
    w = np.copy(X[0,:])
    h = np.copy(X[1,:])

    # second, create new data matrix Z by stacking h and w
    Z = np.vstack((h,w))

    # third, plot this new representation of the data
    plotData2D(Z, path="latex/outliers2.tex")
