#!/usr/bin/python
# encoding: utf8

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from kmeans import KMeans


def measure_time(func):
    """add time measure decorator to the functions"""
    def func_wrapper(*args, **kwargs):
        start_time = time.time()
        a = func(*args, **kwargs)
        end_time = time.time()
        #print("time in seconds: " + str(end_time-start_time))
        return end_time - start_time
    return func_wrapper

dataset = np.genfromtxt("data/data-clustering-1.csv", dtype=float, delimiter=',')

# initialize centroids
dataset = dataset.T
np.random.shuffle(dataset)
#dataset = np.array([[5, 5], [10, 5], [5, 10], [10, 10]])


@measure_time
def lloyd(n):
    kmeans = KMeans(dataset, method="lloyd", centroids=False, k=n, distance="euclidean")
    while not kmeans.convergence:
        kmeans.next()


@measure_time
def macqueen(n):
    kmeans = KMeans(dataset, method="macqueen", k=n, distance="euclidean")
    while not kmeans.convergence:
        kmeans.next()


@measure_time
def hartigan(n):
    kmeans = KMeans(dataset, method="hartigan", k=n, distance="euclidean")
    while not kmeans.convergence:
        kmeans.next()

ks = [2, 3, 4, 5]

lloyd_times = {i:[] for i in ks}
macqueen_times = {i:[] for i in ks}
hartigan_times = {i:[] for i in ks}

for k in ks:
    for i in range(10):
        lloyd_times[k].append(lloyd(k))
        macqueen_times[k].append(macqueen(k))
        hartigan_times[k].append(hartigan(k))

for k in lloyd_times:
    lloyd_times[k] = np.mean(lloyd_times[k])
lloyd_times = list(lloyd_times.values())

for k in macqueen_times:
    macqueen_times[k] = np.mean(macqueen_times[k])
macqueen_times = list(macqueen_times.values())

for k in hartigan_times:
    hartigan_times[k] = np.mean(hartigan_times[k])
hartigan_times = list(hartigan_times.values())

# plot
# check the logarithmic scale
lloyd_times = [np.log(y) * 1000 for y in lloyd_times]
macqueen_times = [np.log(y) * 1000 for y in macqueen_times]
hartigan_times = [np.log(y) * 1000 for y in hartigan_times]

fig = plt.figure(figsize=(7.5, 3.5))
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.plot(lloyd_times, 'r', lw=2, label=r'Lloyd')
ax.plot(macqueen_times, 'b', lw=2,  label=r'MacQueen')
ax.plot(hartigan_times, 'g', lw=2,  label=r'Hartigan')

ax.set_xlabel(r'K means', size=14)
ax.set_ylabel(r'Time', size=14)

plt.legend(loc='upper left')

plt.xticks(range(len(ks)), ks, rotation="vertical")
#path = "latex/timing_kmeans.tex"
#tikz_save(path)
plt.savefig("out/plot_times.png", bbox_inches='tight')
plt.show()
