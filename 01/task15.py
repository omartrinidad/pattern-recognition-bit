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


def plotear(x, y, extra=True):
    """
    """
    fig = plt.figure(figsize=(7.5, 3.5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot(
            x,
            y,
            'ro', lw=2)
    ax.set_xlabel(r'1 / s', size=14)
    ax.set_ylabel(r'n', size=14)

    # log scale
    if extra:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.show()


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


# read images
tree2 = misc.imread("images/tree-2.png")
light3 = misc.imread("images/lightning-3.png", flatten=True)
light0 = misc.imread("images/light.ppm", flatten=True).astype(np.float)

image = light0
image_bin = foreground2BinImg(image)
image_bin = np.logical_not(image_bin)

# ToDo: complete the third step
ns = np.linspace(1, 7, 7).astype(np.int)
scaling_factors = 2 ** ns
boxes_by_scale = []

for scale in scaling_factors:
    boxes, count = box_counting(image_bin, n=scale)
    boxes_by_scale.append(count)
    #misc.imshow(boxes)
    misc.imsave("out/light3_boxes_{}.png".format(count), boxes)

# plot results
plotear(scaling_factors, boxes_by_scale)
plotear(np.log10(scaling_factors), np.log10(boxes_by_scale), extra=False)
