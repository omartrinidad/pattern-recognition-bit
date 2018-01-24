#!/usr/bin/python
# encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg.linalg as lg

from matplotlib2tikz import save as tikz_save
from matplotlib.animation import FuncAnimation


def unison_shuffled(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return zip(a[p], b[p])


def non_monotone(x, w, theta):
    wx = np.dot(w.T, x) - theta
    exp = np.exp(-0.5 * np.square(wx))
    return 2 * exp - 1


# load data
X = np.genfromtxt("data/xor-X.csv", dtype=float, delimiter=',').T
Y = np.genfromtxt("data/xor-y.csv", dtype=float, delimiter=',')

n_examples = X.shape[0]

# initialize weights, and theta, and learning rate
theta = np.random.uniform(low=-0.75, high=0.75)
w = np.random.uniform(low=-0.75, high=0.75, size=(2))
learning_rate = 0.00125
eta_w = 0.005
eta_theta = 0.001

for e in range(51):

    upd_theta = 0
    upd_weight = 0

    # random batch, size 5
    # for x, y in unison_shuffled(X, Y):
    for i in np.arange(n_examples):
        x = X[i]
        y = Y[i]

        yhat = non_monotone(x, w, theta)
        dis = yhat - y

        wx = np.dot(w.T, x) - theta
        exp = np.exp(-0.5 * np.square(wx))
        upd_weight += dis * 2 * exp * wx  * x * -1
        upd_theta += dis * 2 * exp * wx

    w = w - eta_w * upd_weight
    theta = theta - eta_theta * upd_theta

    if e % 2 == 0:
        fig = plt.figure()
        Yhat = [non_monotone(X[i], w, theta) for i in range(len(Y))]
        Yhat = np.where(np.array(Yhat) > 0, 1, -1)

        correct_predictions = np.sum(Yhat == Y)
        plt.title(correct_predictions)
        plt.scatter(X[:,0], X[:,1], c = Yhat);
        plt.show()

        if correct_predictions == 200:
            break
