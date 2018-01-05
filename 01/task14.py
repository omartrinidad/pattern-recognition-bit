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

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # draw quadrants
    y1 =  (1 - np.abs(q1) ** p) ** (1.0/p)
    y2 =  (1 - np.abs(q2) ** p) ** (1.0/p)
    y3 = -(1 - np.abs(q3) ** p) ** (1.0/p)
    y4 = -(1 - np.abs(q4) ** p) ** (1.0/p)

    x = np.concatenate(( q1, q2, q3, q4))
    y = np.concatenate(( y1, y2, y3, y4))

    plt.plot(x, y, label = "p = " + str(p))
    plt.legend(loc='upper right')
    # plt.savefig("out/unityball.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
    # plt.close()
    return plt


unitball(0.5, path="latex/unit_ball.tex")


# By picking any two vectors; i.e. (x1, y1) = (0, 1) and (x2, y2) = (1, 0). Then
# by calculating there norms,  we will find that this will be equal to 1.
p = 0.5
v1_x = 0
v1_y = 1
print("Norm of v1: ", (np.abs(v1_x) ** p + np.abs(v1_y) ** p) ** (1.0/p))
v2_x = 1
v2_y = 0
print("Norm of v2: ", (np.abs(v2_x) ** p + np.abs(v2_y) ** p) ** (1.0/p))
# But by calculating the norm of v1 + v2 = (0, 1) + (1, 0) = (1, 1); This will
# be equal 4.
v3_x = 1
v3_y = 1
print("Norm of (v1 + v2): ", (np.abs(v3_x) ** p + np.abs(v3_y) ** p) ** (1.0/p))
# This result is larger than the summation of norm(v1) and norm(v2) and does not
# satisfy the property of norm for: norm(v1 + v2) <= norm(v1) + norm(v2).
# As a conclusion, this is not really a norm.
