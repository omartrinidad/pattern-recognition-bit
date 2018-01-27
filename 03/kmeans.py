#!/usr/bin/python
# encoding: utf8

"""
ToDos:
- Fix the error line 181, it seems that some clusters have zero elements
- Improve running times
    - Write alternative and efficient implementation
- Improve the timing script
- Improve the animation script
- There is an error in the line 177: total = total/self.dataset[self.clusters[c]].shape[0]
"""


from numpy import genfromtxt, where, random, savetxt
import numpy as np
from math import sqrt
from pprint import pprint


def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance of two vectors
    """
    axis = 0 if b.ndim == 1 else 1
    summation = np.sum((a - b) ** 2, axis=axis)

    return np.sqrt(summation)


class KMeans(object):
    """
    Kmeans implementation
    """

    def __init__(self, dataset, method="lloyd", centroids=False, k=3, distance="euclidean"):
        """
        """

        # flag variable to control the convergence
        self.method = method
        self.convergence = False
        self.dataset = dataset
        self.no_dimensions = self.dataset.shape[1]
        self.k = k
        self.clusters = {i:[] for i in range(self.k)}
        self.centroids = np.zeros((self.k, self.no_dimensions))
        # macqueen vars
        self.i = 0
        self.counters = np.ones((self.k))

        # Distances. ToDo: Include more distances
        if distance == "euclidean":
            self.distance = euclidean_distance
        else:
            self.distance = euclidean_distance

        # Methods
        if method == "macqueen":
            self.next = self.macqueen

        elif method == "lloyd" and not centroids:
            self.next = self.lloyd
            self.random_centroids()
        elif method == "hartigan":
            self.epsilon = 0.1
            self.next = self.hartigan
            self.N = self.dataset.shape[0]
            self.x = np.random.choice(self.k, self.N)
            self.clusters = {i: np.where(self.x == i)[0].tolist() for i in range(self.k)}
            # calculate the centroids
            for c in self.clusters:
                indexes = self.clusters[c]
                self.centroids[c] = np.mean(dataset[indexes], 0)
            self.centroids_before = np.copy(self.centroids)
        else:
            # this depends on the number of columns in the dataset
            # it has something to do with lloyd ¬¬'
            self.k = k
            pass


    def random_centroids(self):
        """
        Create centroids
        """
        no_dimensions = self.dataset.shape[1]

        for i in range(self.k):
            for j, d in enumerate(self.dataset.transpose()):
                mini, maxi = min(d), max(d)
                self.centroids[i, j] = np.random.uniform(mini, maxi)


    def purity_function(self):
        """
        """

        intersect = lambda x, y: list(set(x) & set(y))

        # This values were given arbitrarily
        Ks = {
                0: range(50),
                1: range(50, 100),
                2: range(100, 150)
             }

        N = self.dataset.shape[0]

        total = 0
        for k in Ks:
            maxi = 0
            for c in self.clusters:
                intersection = intersect(Ks[k], self.clusters[c])
                if len(intersection) > maxi:
                    maxi = len(intersection)

            total += maxi

        purity_value = 1.0/N * total
        return purity_value


    def macqueen(self):
        """"""

        if self.i == self.dataset.shape[0]:
            self.convergence = True

        if not self.convergence:

            d = self.dataset[self.i]

            if self.i < self.k:
                self.centroids[self.i] = d
                self.clusters[self.i].append(self.i)
                self.i += 1
            else:
                # determine the winner centroid
                distances = euclidean_distance(d, self.centroids)
                n = np.argmin(distances)

                # recalculate centroids
                self.centroids[n] = self.centroids[n] + 1/(self.counters[n] + 1) * (d - self.centroids[n])

                self.counters[n] += 1
                self.clusters[n].append(self.i)
                self.i += 1


    def lloyd(self):
        """
        The classic kmeans
        """

        if not self.convergence:

            # assign each instance to closest center
            clusters = {i:[] for i in range(self.k)}

            for i, element in enumerate(self.dataset):
                closest = []
                for centroid in self.centroids:
                    closest.append(self.distance(centroid, element))

                # dirty code +_+
                index = closest.index(min(closest))
                clusters[index].append(i)

            self.clusters = clusters

            # recalculate centroids
            old_centroids = self.centroids.copy()

            for c in self.clusters:
                # get the sum of each column (dimension)
                total = np.sum(self.dataset[self.clusters[c]], axis=0)
                print(self.dataset[self.clusters[c]].shape[0])
                total = total/self.dataset[self.clusters[c]].shape[0]
                # update the centroids
                self.centroids[c] = total

            # expensive :O
            comparison = old_centroids == self.centroids
            self.convergence = np.sum(comparison) == comparison.size


    def hartigan(self):
        """"""

        if not self.convergence:
            self.convergence = False

            d = self.dataset[self.i]
            distances = euclidean_distance(d, self.centroids)
            n = np.argmin(distances)

            # if winner != current, calculate the new centroid
            if n != self.x[self.i]:
                self.x[self.i] = n

                # recalculate the centroids
                self.clusters = {i: np.where(self.x == i)[0].tolist() for i in range(self.k)}

                for c in self.clusters:
                    indexes = self.clusters[c]
                    self.centroids[c] = np.mean(self.dataset[indexes], 0)
                # self.i += 0

            self.i += 1

            # finish iteration, implement convergence with epsilon value
            if self.i == self.dataset.shape[0]:
                if np.array_equal(self.centroids_before, self.centroids):
                    self.convergence = True
                else:
                    self.i = 0
                    self.centroids_before = np.copy(self.centroids)
