#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:56:02 2018

@author: Vesal Mohammadi
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
def normalization(data):
    mean=np.mean(data,axis=0)
    ndata=data-mean[np.newaxis,:]
    var=np.var(data,axis=0)
    ndata=ndata/var[np.newaxis,:]
    assert data.shape==ndata.shape
    return ndata
def loadData():
    data = np.loadtxt('data-dimred-X.csv',dtype=np.float,comments='#',delimiter=', ')
    data=data.T
    
    labels = np.loadtxt('data-dimred-y.csv',dtype=np.float,comments='#',delimiter=', ')
    assert data.shape[0]==len(labels)    
    labels=labels.astype(int)    
    ndata=normalization(data)
    return ndata,labels
def calcPCA(ndata):
    covarr = np.cov(ndata.T)
    print(covarr)
    w,v= np.linalg.eigh(covarr)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    pcaV=np.c_[v[:,0],v[:,1],v[:,2]]
    pca2P=np.dot(ndata,pcaV[:,0:2])
    pca3P=np.dot(ndata,pcaV)
    return pca2P.real,pca3P.real

def calcLDA(ndata,labels):
    ul=[1,2,3]
    means=[]
    S=[]
    M=np.mean(ndata,axis=0)
    SB=np.zeros((ndata.shape[1],ndata.shape[1]))
    SW=np.zeros((ndata.shape[1],ndata.shape[1]))
    for i,l in enumerate(ul):
        tmpData=ndata[labels==l]
        means.append(np.mean(tmpData,axis=0))
        # calculate between class covariance matrix
        SB+=np.outer((means[i]-M),(means[i]-M))
        # calculate within class covariance matrix
        si=np.zeros((ndata.shape[1],ndata.shape[1]))
        
        for xi in tmpData:
            si+=np.outer((xi-means[i]),(xi-means[i]))
        S.append(si)
        SW+=(1.0/tmpData.shape[0])*si
    print(SW.shape,SB.shape)
    SWB=np.dot(np.linalg.inv(SW),SB)
    print(SWB.shape)
    w,v= np.linalg.eig(SWB)
    #sort by eigen values
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    ldaV=np.c_[v[:,0],v[:,1],v[:,2]]
    lda2P=np.dot(ndata,ldaV[:,0:2])
    lda3P=np.dot(ndata,ldaV)
    return lda2P.real,lda3P.real

def showPLT(ndata,labels,cmd='PCA'):
    if cmd=='PCA':
        p2,p3=calcPCA(ndata)
    elif cmd=='LDA':
        p2,p3=calcLDA(ndata,labels)
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    
    ax.scatter(p2[labels==1][:,0], p2[labels==1][:,1], color= 'b',  marker='o',  label='class 1')
    ax.scatter(p2[labels==2][:,0], p2[labels==2][:,1], color= 'r',  marker='o', label='class 2')
    ax.scatter(p2[labels==3][:,0], p2[labels==3][:,1], color= 'g',  marker='o', label='class 3')
    ax.legend(loc='upper right')
    plt.title(cmd+ ' projection into 2d space')
   
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(p3[labels==1][:,0], p3[labels==1][:,1],p3[labels==1][:,2], color= 'b',  marker='o',  label='class 1')
    ax.scatter(p3[labels==2][:,0], p3[labels==2][:,1],p3[labels==2][:,2], color= 'r',  marker='o', label='class 2')
    ax.scatter(p3[labels==3][:,0], p3[labels==3][:,1],p3[labels==3][:,2], color= 'g',  marker='o', label='class 3')
    ax.legend(loc='upper left')
    plt.title(cmd+ 'projection into 3d space')
    plt.show()
if __name__ == "__main__":
    ndata,labels=loadData()
    showPLT(ndata,labels,'PCA')
    showPLT(ndata,labels,'LDA')
    
    
    
