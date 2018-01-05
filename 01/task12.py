#!/usr/bin/env python
"""
task 1.2 Fitting a Normal distribution to 1D data
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from auxiliar import *


@save_figure_latex
def plot(hs, mean, std):
    """
    # ToDo: hack the tails
    """
    fig, ax = plt.subplots()
    plt.hist(hs, normed=True, alpha=0.5, bins=4)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin - 25, xmax + 25, 100)

    pdf = norm.pdf(x, mean, std)
    #pdf = np.hstack([0, pdf, 0])

    plt.plot(x, pdf, 'blue', linewidth=2)
    plt.plot(hs, len(hs) * [0.002], 'o', markersize=6, color='grey')

    # fix x and y labels
    y = plt.yticks()[0]
    plt.yticks(y, y)

    plt.xlabel("Height")
    plt.ylabel("Bins")

    # plt.show()

    return plt


dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
data = np.loadtxt('data/whData.dat', dtype=dt, comments='#', delimiter=None)

hs = np.array([d[1] for d in data])

mean = np.mean(hs)
std = np.std(hs)

plot(hs, mean, std)
