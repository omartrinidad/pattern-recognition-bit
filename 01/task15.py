#!/usr/bin/env python

"""
Task 1.5 Estimating the dimension of fractal objects in an image.
"""

import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as img
import numpy.linalg as la

from scipy import misc
from scipy.stats import linregress
from auxiliar import *


def data_matrix_V1(x):
    """ Taken from Recipes for Data Science """
    n = len(x)
    return np.vstack((x, np.ones(n))).T


def lsq_solution_V1(X, y):
    """ Taken from Recipes for Data Science """
    w = np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), y)
    return w


def least_squares(x, y):
    """
    Given some points return a linear model
    """
    # x becomes X
    X = data_matrix_V1(x)
    w = lsq_solution_V1(X, y)
    return w


def foreground2BinImg(f):
    """
    Binarization given by professor Bauckhage
    """
    d = img.filters.gaussian_filter(f, sigma=0.50, mode="reflect") - \
        img.filters.gaussian_filter(f, sigma=1.00, mode="reflect")
    d = np.abs(d)
    m = d.max()
    d[d< 0.1*m] = 0
    d[d>=0.1*m] = 1
    return img.morphology.binary_closing(d)


@save_figure()
def plot_both_models(image1, image2, path="", extra=True, save=False, show=False, title=""):
    """
    Function that plots two models and make a comparison
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # first model
    scaling_factors, boxes_by_scale = generate_images(image1, path, show=False)
    x = np.log(scaling_factors)
    y = np.log(boxes_by_scale)

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot( x, y, 'ro', lw=3, label="Tree")

    # fit a line
    lr = linregress(x, y)
    slope = lr.slope # D, slope
    offset = lr.intercept # b, offset

    abline_values = [slope * i + offset for i in range(0, 7)]
    ax.plot(range(0, 7), abline_values, color="r")
    
    # second model
    scaling_factors, boxes_by_scale = generate_images(image2, path, show=False)
    x = np.log(scaling_factors)
    y = np.log(boxes_by_scale)

    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot( x, y, 'bo', lw=3, label="Light")

    # fit a line
    lr = linregress(x, y)
    slope = lr.slope # D, slope
    offset = lr.intercept # b, offset

    abline_values = [slope * i + offset for i in range(0, 7)]
    ax.plot(range(0, 7), abline_values, color="b")

    ax.set_xlabel(r'$\log{\frac{1}{s}}$', size=14)
    ax.set_ylabel(r'$\log{n}$', size=14)

    # log scale
    #if extra:
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')

    plt.legend()
    #if save:
    #    plt.savefig(path + "both_plots.png", bbox_inches="tight", pad_inches=0)
    #if show:
    #    plt.show()

    return plt


def plotting(x, y, path, extra=True, save=False, show=False, title=""):
    """
    x, y, log values
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot( x, y, 'ro', lw=2)

    # fit a line
    lr = linregress(x, y)
    slope = lr.slope # D, slope
    offset = lr.intercept # b, offset

    plt.title(" {}\nSlope {:0.3f},\nOffset {:0.3f}".format(title, slope, offset))

    abline_values = [slope * i + offset for i in range(0, 7)]
    ax.plot(range(0, 7), abline_values)

    ax.set_xlabel(r'$\log{\frac{1}{s}}$', size=14)
    ax.set_ylabel(r'$\log{n}$', size=14)

    # log scale
    if extra:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if save:
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()


def draw_rectangle(image, x, y, size):
    """
    Given a black-and-white image with 3 channels, draw a rectancle in specific
    coordinates.
    """
    # draw horizontal lines
    image[:,:,0][x:x+1, y:y+size] = 0
    image[:,:,1][x:x+1, y:y+size] = 0 
    image[:,:,2][x:x+1, y:y+size] = 1
    #
    image[:,:,0][x+size:x+size+1, y:y+size] = 0
    image[:,:,1][x+size:x+size+1, y:y+size] = 0
    image[:,:,2][x+size:x+size+1, y:y+size] = 1
    # draw vertical lines
    image[:,:,0][x:x+size, y:y+1] = 0
    image[:,:,1][x:x+size, y:y+1] = 0 
    image[:,:,2][x:x+size, y:y+1] = 1
    #
    image[:,:,0][x:x+size, y+size:y+size+1] = 0
    image[:,:,1][x:x+size, y+size:y+size+1] = 0
    image[:,:,2][x:x+size, y+size:y+size+1] = 1


def box_counting(image_bin, n=2):
    """
    """

    boxes = np.zeros((512, 512, 3))
    boxes[:,:,0] = image_bin
    boxes[:,:,1] = image_bin
    boxes[:,:,2] = image_bin

    points = np.linspace(0, image_bin.shape[0], n, endpoint=False)
    points = points.astype(np.integer)
    step = points[1]

    # ACHTUNG: vectorize this part!
    patches = np.zeros((n * n))
    a = 0
    for p in points:
        for q in points:
            # if the patch is white
            if np.sum(image_bin[p:p+step, q:q+step]) != step ** 2:
                patches[a] = 1
            a += 1

    patches = patches.reshape((n, n))
    no_boxes = int(np.sum(patches))
    patches_pos = np.argwhere(patches == 1) * step

    # draw patches
    for pair in patches_pos:
        x, y = pair[0], pair[1]
        draw_rectangle(boxes, x, y, step)

    return boxes, no_boxes


def generate_images(image, path, show=True, save=False):
    """
    """
    image_bin = foreground2BinImg(image)
    image_bin = np.logical_not(image_bin)

    misc.imsave(path + "_binary.png", image_bin)

    ns = np.linspace(1, 7, 7).astype(np.int)
    scaling_factors = 2 ** ns
    boxes_by_scale = []

    for scale in scaling_factors:
        boxes, count = box_counting(image_bin, n=scale)
        boxes_by_scale.append(count)

        if save:
            misc.imsave(path + "_{}.png".format(count), scale)
        if show:
            misc.imshow(boxes)

    return scaling_factors, boxes_by_scale


# read images
light0 = misc.imread("images/light.ppm", flatten=True).astype(np.float)
tree2 = misc.imread("images/tree-2.png", flatten=True).astype(np.float)
light3 = misc.imread("images/lightning-3.png", flatten=True)

# Solution for exemplary image
path = 'out/light/boxes'
scaling_factors, boxes_by_scale = generate_images(light0, path, show=True)
log_1_si = np.log(scaling_factors)
log_ni = np.log(boxes_by_scale)
plotting(log_1_si, log_ni, path + "_points.png", extra=False, show=True)

# Solution for tree-2.png image
path = 'out/tree2/boxes'
scaling_factors, boxes_by_scale = generate_images(tree2, path, show=True)
log_1_si = np.log(scaling_factors)
log_ni = np.log(boxes_by_scale)
plotting(log_1_si, log_ni, path + "_points.png", extra=False, show=True, title="Tree")

# Solution for lightning-3.png image
path = 'out/light3/boxes'
scaling_factors, boxes_by_scale = generate_images(light3, path, show=True)
log_1_si = np.log(scaling_factors)
log_ni = np.log(boxes_by_scale)
plotting(log_1_si, log_ni, path + "_points.png", extra=False, show=True, title="Light")

# Plot to compare both models
path = 'out/'

plot_both_models(tree2, light3, path="latex/both.tex", extra=True, save=False, show=True, title="")
