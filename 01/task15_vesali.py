#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:31:02 2017

@author: vmohammadi
"""
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import math
def boxing(w,h,m,n):
    wrange=np.arange(0,w+1,m)
    wrange=wrange.tolist()
    if wrange[-1]<w:
        wrange.append(w)
    hrange=np.arange(0,h+1,n)
    hrange=hrange.tolist()
    if hrange[-1]<h:
        hrange.append(h)
    myList=[]
    for j,th in enumerate(hrange[:-1]):
        for i,tw in enumerate(wrange[:-1]):
            #print('{}:{},{}:{}'.format(th,hrange[j+1],tw,wrange[i+1]))
            myList.append([th,hrange[j+1],tw,wrange[i+1]])
    return myList
def foreground2BinImg(f):
    d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - img.filters.gaussian_filter(f, sigma=1.00, mode='reflect')
    d = np.abs(d)
    m = d.max()
    d[d< 0.1*m] = 0
    d[d>=0.1*m] = 1
    return img.morphology.binary_closing(d)
#imgName = 'lightning-3'
imgName = 'tree-2'
f = msc.imread(imgName+'.png', flatten=True).astype(np.float)
myImg = foreground2BinImg(f)
H ,W = img.imread(imgName+'.png').shape
print (H)
sList=[1/(2**i) for i in range(1,int(math.log2(H))-2)]
print(sList)
#sList=sList[0:6]
plR=int(len(sList)/3)+1
xPlot=[]
yPlot=[]
for sIdx,s in enumerate(sList):
    xPlot.append(math.log10(1/s))
    g=np.copy(myImg)
    m=int(W*s)
    n=int(H*s)
    indices=boxing(W,H,m,n)
    mask=[0]*len(indices)
    for i,ind in enumerate(indices):
        if np.max(g[ind[0]:ind[1],ind[2]:ind[3]])>=1:
            mask[i]=1
    yPlot.append(math.log10(np.sum(mask)))
    for i,m in enumerate(mask):
        if m==1:
            #print(indices[i])
            
            g[indices[i][0],indices[i][2]:indices[i][3]-1]=1
            g[indices[i][1]-1,indices[i][2]:indices[i][3]-1]=1
            g[indices[i][0]:indices[i][1]-1,indices[i][2]]=1
            g[indices[i][0]:indices[i][1]-1,indices[i][3]-1]=1
    ax=plt.subplot(plR,3,sIdx+1)
    ax.imshow(g, cmap='Greys')
ax=plt.subplot(3,3,8)
ax.scatter(xPlot, yPlot)
plt.show()
