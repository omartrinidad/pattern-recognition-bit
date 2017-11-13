#!/usr/bin/env python

"""
Task 1.5 Estimating the dimension of fractal objects in an image.
"""

import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as img
from scipy import misc


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


def draw_rectangle(image, x, y, size):
    """
    Given a black-and-white image with 3 channels, draw a rectancle in specific
    coordinates.
    """
    # draw horizontal lines
    image[:,:,0][x:x+1, y:y+size] = 1
    image[:,:,1][x:x+1, y:y+size] = 0
    image[:,:,2][x:x+1, y:y+size] = 0
    #
    image[:,:,0][x+size:x+size+1, y:y+size] = 1
    image[:,:,1][x+size:x+size+1, y:y+size] = 0
    image[:,:,2][x+size:x+size+1, y:y+size] = 0
    # draw vertical lines
    image[:,:,0][x:x+size, y:y+1] = 1
    image[:,:,1][x:x+size, y:y+1] = 0
    image[:,:,2][x:x+size, y:y+1] = 0
    #
    image[:,:,0][x:x+size, y+size:y+size+1] = 1
    image[:,:,1][x:x+size, y+size:y+size+1] = 0
    image[:,:,2][x:x+size, y+size:y+size+1] = 0


def binarization(f, t=0):
    return np.where(f>=t, 1, 0)


def binarization2(f, t=0):
    return np.where(f>=t, 0, 1)


def draw_boxes(image, n=2):
    """
    """
    points = np.linspace(0, image.shape[0], 3)


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
    no_boxes = np.sum(patches)
    patches_pos = np.argwhere(patches == 1) * step

    # draw patches
    for pair in patches_pos:
        x, y = pair[0], pair[1]
        draw_rectangle(boxes, x, y, step)

    return boxes, no_boxes


# read images
#tree2 = misc.imread("tree-2.png")
#light3 = misc.imread("lightning-3.png", flatten=True)
imgName = "images/light.ppm"
f = misc.imread(imgName, flatten=True).astype(np.float)
image_bin = foreground2BinImg(f)
image_bin = np.logical_not(image_bin)

# ToDo: complete the third step
# apply box counting
_, count = box_counting(image_bin, n=4)
_, count = box_counting(image_bin, n=8)
boxes, count = box_counting(image_bin, n=16)
#misc.imshow(boxes)
