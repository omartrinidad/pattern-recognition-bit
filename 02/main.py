#!/usr/bin/python
# encoding: utf8
import numpy as np
import matplotlib.pyplot as plt
from knn import *
from auxiliar import *

# ToDo: Add Voronoi diagram limits

@save_figure()
def scatterplot(dataset, path=""):
    fig = plt.figure(figsize=(7.5, 3.5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
            loc=3, ncol=1, mode="expand", borderaxespad=0., fontsize=12)

    negative = np.where(dataset.training_labels == -1)
    x = dataset.training[negative,0:1]
    y = dataset.training[negative,1:2]
    neg = plt.scatter(x, y, marker="o", label="Negative")

    positive = np.where(dataset.training_labels == 1)
    x = dataset.training[positive,0:1]
    y = dataset.training[positive,1:2]
    pos = plt.scatter(x, y, marker="^", label="Positive")

    # plt.xticks(range(len(xticks)), xticks)
    plt.legend(loc='upper left')
    return plt


@save_figure()
def performance_plot(
        times, colors, labels,
        xlabel, ylabel, xticks,
        log_scale=False,
        path=""
        ):
    """
    ToDo: 
    - Check the logarithmic scale.
    """

    if log_scale:
        for i, t in enumerate(times):
            times[i] = [np.log(y) * 1000 for y in t]

    fig = plt.figure(figsize=(7.5, 3.5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    for t, color, label in zip(times, colors, labels):
        ax.plot(t, color, lw=2, label=label)

    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
            loc=3, ncol=1, mode="expand", borderaxespad=0., fontsize=12)

    plt.xticks(range(len(xticks)), xticks)
    return plt


def evaluation(pred_labels, test_labels):
    """
    Calculate accuracy
    """
    for ik in pred_labels:
        msg = "{0} -> ks, {1:0.02f} of tests examples classified correctly".format(
                ik, np.mean(pred_labels[ik] == test_labels) * 100
                )
        print(msg)


training = np.genfromtxt("data/data2-train.dat", dtype=float, delimiter=' ')
test = np.genfromtxt("data/data2-test.dat", dtype=float, delimiter=' ')

dataset = Dataset(training, test)

myknn = kNN(dataset)

# Evaluation with euclidean distance
ks = [1, 5, 7, 9, 10, 15]
times1 = list()
for k in ks:
    time = myknn.fit(k = [k], metric="euclidean")
    times1.append(time)

#print("Running kNN with k = 1, 3, 5 and 7 using Euclidean distance")
#evaluation(myknn.pred_labels, myknn.test_labels)

# Evaluation with cosine similarity
times2 = list()
for k in ks:
    time = myknn.fit(k = [k], metric="cosine")
    times2.append(time)

myknn.normalization()

times3 = list()
for k in ks:
    time = myknn.fit(k = [k], metric="euclidean")
    times3.append(time)

times4 = list()
for k in ks:
    time = myknn.fit(k = [k], metric="cosine")
    times4.append(time)

scatterplot(dataset, path="latex/knn.tex")

colors = ['r', 'b', 'g', 'black']
labels = ['Euclidean', 'Cosine', 'Euclidean normalized', 'Cosine normalized']
times = [times1, times2, times3, times4]
performance_plot(times, colors, labels, "Nearest neighbors", "Miliseconds", ks, log_scale=False, path="out/knn_times.png")
