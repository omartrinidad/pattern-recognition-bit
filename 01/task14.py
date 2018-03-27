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
from auxiliar import *


@save_figure()
def unitball(p, path=""):
    """
    """

    # x points created in the order of quadrants
    q1 = np.linspace(1.0, 0.0, 1000)
    q2 = np.linspace(0.0, -1.0, 1000)
    q3 = q2[::-1]
    q4 = q1[::-1]

    fig, ax = plt.subplots()
    plt.xlim(-1.125, 1.125)
    plt.ylim(-1.125, 1.125)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # draw quadrants
    y1 =  (1 - np.abs(q1) ** p) ** (1.0/p)
    y2 =  (1 - np.abs(q2) ** p) ** (1.0/p)
    y3 = -(1 - np.abs(q3) ** p) ** (1.0/p)
    y4 = -(1 - np.abs(q4) ** p) ** (1.0/p)

    x = np.concatenate(( q1, q2, q3, q4))
    y = np.concatenate(( y1, y2, y3, y4))

    ax.set_facecolor("#eeeeff")
    plt.plot(x, y, label = "p = " + str(p))
    # plt.savefig("out/unityball.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
    # plt.close()
    return plt


unitball(0.5, path="latex/unit_ball.tex")
