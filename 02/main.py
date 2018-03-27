#!/usr/bin/python
# encoding: utf8
import numpy as np
import matplotlib.pyplot as plt
import timeit
from knn import *
from auxiliar import *


@save_figure()
def scatterplot(dataset, labels, path=""):
    """
    """

    fig, ax = plt.subplots()

    labels = np.array(labels)

    negative = np.where(labels == -1)[0]
    x = dataset.test[negative,0:1]
    y = dataset.test[negative,1:2]
    plt.scatter(x, y, marker="o", label="Negative", alpha=0.666, c="#2222ee")

    positive = np.where(labels == 1)[0]
    x = dataset.test[positive,0:1]
    y = dataset.test[positive,1:2]
    plt.scatter(x, y, marker="o", label="Positive", alpha=0.333, c="#ee2222")

    # plt.xticks(range(len(xticks)), xticks)
    plt.legend(loc='upper left')
    ax.set_facecolor("#ffffe0")

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

    fig = plt.figure(figsize=(7.5, 0.5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    for t, color, label in zip(times, colors, labels):
        ax.plot(t, color, lw=2, label=label)

    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)

    plt.legend(loc='upper left')
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
ks = [1, 3, 5, 7, 9, 11]

times1 = list()
for k in ks:
    time = myknn.fit(k = [k], metric="euclidean")
    times1.append(time)

#print("Running kNN with k = 1, 3, 5 and 7 using Euclidean distance")
#myknn.normalization()

times3 = list()
for k in ks:
    time = myknn.fit(k = [k], metric="euclidean")
    times3.append(time)

evaluation(myknn.pred_labels, myknn.test_labels)
scatterplot(dataset, myknn.pred_labels[1], path="latex/knn1.tex")
scatterplot(dataset, myknn.pred_labels[5], path="latex/knn5.tex")
scatterplot(dataset, myknn.pred_labels[11], path="latex/knn11.tex")

"""
colors = ['r', 'g' ]
labels = ['Euclidean', 'Euclidean normalized']
times = [times1, times3]
performance_plot(
        times, colors, labels, "Nearest neighbors",
        "Miliseconds", ks, log_scale=False, path="latex/knn_times.tex"
        )
"""
