#!/usr/bin/python
# encoding: utf8

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from kmeans import KMeans
from auxiliar import *


@save_figure()
def plot_clusters(kmeans, path=""):
    """
    """
    fig, ax = plt.subplots()
    colors = ["#2222ee", "#ee2222", "#00aa00"]

    for v, color, n in zip(kmeans.clusters.values(), colors, [1, 2, 3]):
        labels = np.array(v)
        x = dataset[labels,0:1]
        y = dataset[labels,1:2]
        plt.scatter(x, y, marker="o", label="Cluster {}".format(n), alpha=0.2, c=color)


    colors = ["#0000ff", "#ff0000", "#006600"]
    for v, color in zip(kmeans.centroids, colors):
        plt.scatter(v[0], v[1], marker="H", c=color, s=111)


    ax.set_facecolor("#ffffee")
    plt.legend(loc='upper left')
    return plt


dataset = np.genfromtxt("data/data-clustering-1.csv", dtype=float, delimiter=',')

# initialize centroids
dataset = dataset.T
np.random.shuffle(dataset)
#dataset = np.array([[5, 5], [10, 5], [5, 10], [10, 10]])

kmeans = KMeans(dataset, method="lloyd", centroids=False, k=3, distance="euclidean")
while not kmeans.convergence:
    kmeans.next()
plot_clusters(kmeans, path="latex/lloyd.tex")


kmeans = KMeans(dataset, method="macqueen", k=3, distance="euclidean")
while not kmeans.convergence:
    kmeans.next()
plot_clusters(kmeans, path="latex/macqueen.tex")


kmeans = KMeans(dataset, method="hartigan", k=3, distance="euclidean")
while not kmeans.convergence:
    kmeans.next()
plot_clusters(kmeans, path="latex/hartigan.tex")

