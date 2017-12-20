#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:41:13 2017

@author: vmohammadi
"""

"""
======================================================================
          Task 2.2:Least Squares Regression with Polynomials          
======================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def draw_plot(X,y,nX,ny,mx,my):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.set_xlabel('Weight')
    axs.xaxis.label.set_color('blue')
    axs.set_ylabel('Height')
    axs.yaxis.label.set_color('blue')
    
    # plot the data 
    axs.plot(X[:,1], y, 'ro', label='data')
    axs.plot(nX[:,1], ny, label='Least Squares Regression')
    axs.plot(mx,my, 'ro',c= 'green',  label='Predict outliers')
    
    axs.tick_params(axis='x', labelsize=13)
    axs.tick_params(axis='y', labelsize=13)
    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)
    plt.show()
    
def data_matrix(x):
    
    return np.c_[np.ones(x.shape[0]),x]
def lsq_solution(X, y):
    w = np.linalg.lstsq(X, y)[0]
    return w
########## reading data #########
dt = np.dtype([('weight', np.float), ('height', np.float), ('gender', np.str_, 1)])
data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

w = np.array([d[0] for d in data])
h = np.array([d[1] for d in data])
x=[]
y=[]
for i in range(len(w)):
    if w[i]>0:
        x.append(w[i])
        y.append(h[i])
########## calculate W ############
for d in [1,5,10]:
    X=np.zeros((len(x),d))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j]=x[i]**(j+1)
    X=data_matrix(X)
    W=lsq_solution(X,y)

########### predict missing value ########
    mx=[]
    my=[]
    for i in range(len(w)):
        if w[i]<=0:
            minDif=h[i]
            nw=W[0]
            for j in range(int(X[:,1].min()),int(X[:,1].max())):
                pw=0
                for ik,k in enumerate(W):
                    pw+=k*(j**ik)
                if abs(h[i]-pw)<minDif:
                    minDif=abs(h[i]-pw)
                    nw=j
            mx.append(nw)
            my.append(h[i])
            print  ("weight: " ,nw, "height: ",h[i])
    print("dgree is ", ik)
            
########### producing new data #######
    
    nx=np.linspace(X[:,1].min(),X[:,1].max(),200)
    nX=np.zeros((len(nx),d))
    
    for i in range(nX.shape[0]):
        for j in range(nX.shape[1]):
            nX[i,j]=nx[i]**(j+1)
    nX=data_matrix(nX)
    
    ny=np.dot(nX,W)
    draw_plot(X,y,nX,ny,mx,my)