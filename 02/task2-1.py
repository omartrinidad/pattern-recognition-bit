#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:00:26 2017

@author: vmohammadi
"""

###############################################
####### Task 2.1:Linear Regression   ##########       
###############################################
import numpy as np
import matplotlib.pyplot as plt

def plotData2D(X,Y, filename=None):
    #draw axis 
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.set_xlabel('Weight')
    axs.xaxis.label.set_color('blue')
    axs.set_ylabel('Height')
    axs.yaxis.label.set_color('blue')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')
    axs.plot(Y[0,:], Y[1,:], label='Linear Regression')
    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
	
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(X[1,:].min()-10, X[1,:].max()+10)
    axs.tick_params(axis='x', labelsize=13)
    axs.tick_params(axis='y', labelsize=13)
    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # we can show the plot or save it as file
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


# read data as 2D array of data type 'object'
dt = np.dtype([('weight', np.float), ('height', np.float), ('gender', np.str_, 1)])
data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)
#data=np.sort(data,order='weight')

# create matrix from data
w = np.array([d[0] for d in data])
h = np.array([d[1] for d in data])
x=[]
y=[]
    
#remove outlier points
for i in range(len(w)):
    if w[i]>0:
        x.append(w[i])
        y.append(h[i])
        
weightmatrix = np.vstack([x, np.ones(len(x))]).T
w1, w2 = np.linalg.lstsq(weightmatrix, y)[0]
temp2= np.copy(weightmatrix[:,0])
heightvalue=np.copy(weightmatrix[:,0])*w1 + w2
    
#Predict outliers weight
for i in range(len(w)):
    if w[i]<0:
        print ("Height=%s   Predicted weight:%.3f"%(h[i],(h[i]-w2)/w1))
		 
#Linear Regression
h=np.array(np.copy(weightmatrix[:,0]),np.float)  
w=np.array(heightvalue,np.float)
hw=np.array([h,w],np.float)
	
#input data
h1=np.array(x,np.float)  
w1=np.array(y,np.float)
hw1=np.array([h1,w1],np.float)
		
#Plot 
plotData2D(hw1,hw,'Task02.1.pdf')