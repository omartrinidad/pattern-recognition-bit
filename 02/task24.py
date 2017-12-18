#!/usr/bin/env python
"""
Task 2.4 Boolean functions and the Boolean Fourier transform
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from itertools import chain, combinations
from functools import reduce


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def phi(x):
    """
    Second part of the task
    """
    length = len(x)
    ps = powerset(np.arange(length))
    phi = map(lambda e: reduce(lambda xe, ye: xe * x[ye], e, 1), list(ps))
    return list(phi)


def data_matrix_V1(x):
    n = len(x)
    b = np.ones((n, 1))
    return np.hstack((x, b))


def lsq_solution_V1(X, y):
    w = np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), y)
    return w


def lsq_solution_V3(X, y):
    w, residual, rank, svalues = la.lstsq(X, y)
    return w


def wolfram_rule(number):
    """
    Creates a binary Wolfram Number
    """
    integers = np.array([number]).astype(np.uint8)
    wolfram_number = np.unpackbits(integers)
    #shape = (1, len(wolfram_number))
    #wolfram_number = wolfram_number.reshape(shape)
    return wolfram_number

# First part of the task
# design matrix X, tricky way ;)
integers = np.linspace(7, 0, 8).astype(np.uint8)
bits = np.unpackbits(integers)
bits = bits.reshape((8,8))[:,5:]
bits = np.where(bits == 0, -1, +1)

# y matrix
y_110 = wolfram_rule(110)
y_126 = wolfram_rule(126)

# y_hat matrix
X = data_matrix_V1(bits)
w_110 = lsq_solution_V3(X, y_110)
w_126 = lsq_solution_V3(X, y_126)

#print(w_110)
#print(w_126)

yhat_110 = np.dot(X, w_110)
yhat_126 = np.dot(X, w_126)

print("Comparison between yhat and y. 126")
print(y_126)
print(np.around(yhat_126))

print("Comparison between yhat and y. 110")
print(y_110)
print(np.around(yhat_110))

# Third part of the task
# Generate the PHI matrix
big_phi = phi(bits[0])
big_phi = np.array(big_phi)
big_phi = big_phi.reshape(8, 1)

for b in bits[1:]:
    bi = phi(b)
    bi = np.array(bi)
    bi = bi.reshape(8, 1)
    print(big_phi.shape)
    print(bi.shape)
    big_phi = np.hstack((big_phi, bi))


print("------------------------------------")
print("Comparison between yhat and y. 110")
w_110 = lsq_solution_V3(big_phi, y_110)
yhat_110 = np.dot(big_phi, w_110)
print(y_110)
print(np.around(yhat_110))

print("Comparison between yhat and y. 126")
w_110 = lsq_solution_V3(big_phi, y_110)
w_126 = lsq_solution_V3(big_phi, y_126)
yhat_126 = np.dot(big_phi, w_126)
print(y_126)
print(np.around(yhat_126))
