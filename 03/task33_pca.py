#!/usr/bin/python
# encoding: utf8

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data
X = np.genfromtxt("data/data-dimred-X.csv", dtype=float, delimiter=',').T
y = np.genfromtxt("data/data-dimred-y.csv", dtype=float, delimiter=',')

# zero mean, normalization,
m0 = np.mean(X, axis=0)
# center the data
normalized = np.subtract(X, m0)

# covariance matrix. Why rowvar == normalized.T?
C = np.cov(normalized.T)

# compute eigenvectors and eigenvalues
eival, eivec = np.linalg.eigh(C)

#inds = np.argsort(eival)[::-1]
#eival = eival[inds]
#eivec = eivec[:,inds]
eivec = eivec.T[::-1]

# plots
# plot in 2D
z = np.dot(normalized, eivec[:3].T)
# for plotting
z = z.T 

plt.scatter(z[0], z[1], c=y)
plt.savefig("out/03/pca_2d.png", bbox_inches="tight", pad_inches=0)
plt.show()

# plot in 2D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(z[0], z[1], z[2], c=y);
plt.savefig("out/03/pca_3d.png", bbox_inches="tight", pad_inches=0)
plt.show()
