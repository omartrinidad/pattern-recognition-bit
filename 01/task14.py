#!/usr/bin/env python

"""
task 1.4 Unit circles

# ToDo: Implement it in three dimensions
# L_p norm
# https://www.youtube.com/watch?v=SXEYIGqXSxk
# unit circle:
# https://www.youtube.com/watch?v=qTbDQ9gkKJg
"""

import pylab
import matplotlib.pyplot as plt
import numpy as np


def save_figure(func):
    """add time measure decorator to the functions"""
    def func_wrapper(*args, **kwargs):
        plt = func(*args, **kwargs)
        plt.savefig("out/fit_normal.png", bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.close()
    return func_wrapper


def unitball():
    """
    """

    # x points created in the order of quadrants
    q1 = np.linspace(1.0, 0.0, 100)
    q2 = np.linspace(0.0, -1.0, 100)
    q3 = q2[::-1]
    q4 = q1[::-1]

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    for p in [0.3, 0.5, 1, 2, 4]:
        y1 =  (1 - np.abs(q1) ** p) ** (1/p)
        y2 =  (1 - np.abs(q2) ** p) ** (1/p)
        y3 = -(1 - np.abs(q3) ** p) ** (1/p)
        y4 = -(1 - np.abs(q4) ** p) ** (1/p)

        x = np.concatenate(( q1, q2, q3, q4))
        y = np.concatenate(( y1, y2, y3, y4))

        plt.plot(x, y)

    plt.savefig("out/unityball.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()

unitball()
