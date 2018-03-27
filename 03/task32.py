#!/usr/bin/python
# encoding: utf8


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib2tikz import save as tikz_save

from matplotlib import rc
#rc("text", usetex=True)




def spectral(X, beta):
    """
    """
    # calculate similarity matrix S
    # cdist <--- Euclidean by defaultS
    S = np.exp(-beta * cdist(X, X))

    # Laplacian matrix L = D - S
    L = np.diag(np.sum(S, axis=0)) - S

    # the eigenvalues are returned in ascending order
    # second = np.argsort(eival)[1]
    _, eivec = np.linalg.eigh(L)

    # Get the Fiedler vector, and clusterize
    return np.where(eivec[:, 1] > 0, 1, -1)


X = np.genfromtxt('data/data-clustering-2.csv', dtype=float, delimiter=',').T

for i in [3, 7]:
    clusters = spectral(X, i)
    #path = "out/02/beta{}.png".format(i)
    path="latex/spectral_{}.tex".format(i)
    #plot(X, clusters, "X", path)

    fig = plt.figure()
    x, y = X[:,0], X[:,1]
    a = (clusters == -1).nonzero()
    b = (clusters == 1).nonzero()
    plt.scatter(x[a], y[a], c="#ee2222", label="Cluster 1", marker="s", alpha=0.333)
    plt.scatter(x[b], y[b], c="#2222ee", label="Cluster 2", marker="s", alpha=0.333)

    plt.legend(loc='upper left')

    if path:
        tikz_save(path)

    plt.show()
