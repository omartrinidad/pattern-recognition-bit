#!/usr/bin/python
# encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def plot(X, clusters, title=None):
    """
    """
    clusters = np.where(clusters == 1, "#ee2222", "#2222ee")
    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1], c=clusters, marker="s", alpha=0.333)

    if title:
        plt.title(title)

    plt.show()


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

for i in [1, 3, 5, 7]:
    clusters = spectral(X, i)
    plot(X, clusters, "Beta {}".format(i))

