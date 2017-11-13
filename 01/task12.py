#!/usr/bin/env python
"""
task 1.2 Fitting a Normal distribution to 1D data
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm


def measure_time(func):
    """add time measure decorator to the functions"""
    def func_wrapper(*args, **kwargs):
        start_time = time.time()
        a = func(*args, **kwargs)
        end_time = time.time()
        print("time in seconds: " + str(end_time-start_time))
        return end_time - start_time
    return func_wrapper


def save_figure(func):
    """add time measure decorator to the functions"""
    def func_wrapper(*args, **kwargs):
        plt = func(*args, **kwargs)
        plt.savefig("out/fit_normal.png", bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.close()
    return func_wrapper


@save_figure
def plot(hs, mean, std):
    """
    Plot data
    # ToDo: hack the tails
    """
    plt.hist(hs, normed=True, alpha=0.5, bins=4)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin - 25, xmax + 25, 100)

    pdf = norm.pdf(x, mean, std)
    #pdf = np.hstack([0, pdf, 0])

    plt.plot(x, pdf, 'blue', linewidth=2)
    plt.plot(hs, len(hs) * [0.002], 'o', markersize=12, color='#00ff00')

    #plt.show()
    #plt.close()

    return plt


dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
data = np.loadtxt('data/whData.dat', dtype=dt, comments='#', delimiter=None)

#hs = norm.rvs(loc=0, scale=1, size=200)
hs = np.array([d[1] for d in data])
#hs = norm.rvs(10.0, 2.5, size=500)

mean = np.mean(hs)
std = np.std(hs)

# adjust the mean and the standard deviation
#hs = hs - mean
#hs = hs/std

plot(hs, mean, std)
