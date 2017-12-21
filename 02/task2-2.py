import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import chi2

data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

# read height and weight data into into a 2d matrix
data = data[:,0:2].astype(np.float)

# remove the outliers: training data= traind
trainD = data[data[:,0] >= 0]

# prediction data (outliers): Prediction Data= predD
predD = data[data[:,0] ==-1]

# create height vector for prediction data: PreX,PreY for predictionX and Prediction Y
preX = np.copy(predD[:,1])

# weight vector for train data
y = np.copy(trainD[:,0])
# create height vector for train data
x = np.copy(trainD[:,1])
#calculation!
mw = np.round(np.mean(y),2)
mh = np.round(np.mean(x),2)
devw = np.round(np.std(y),2)
devh = np.round(np.std(x),2)
corrcoef = np.corrcoef(x,y)[0,1]
#
#
z = 0
preY =np.zeros([len(preX)])
while (z!=len(preX)):
    h0 = preX[z]
    prediction = mw+corrcoef *(h0 - mh) *devw/devh
    print(h0)
    print (prediction)
    preY[z] = prediction
    z= z + 1
#plotting

plt.plot(x, y, 'go',preX,preY,'rs')
plt.axis([np.amin(x)-10, np.amax(x)+20, np.amin(y)-10, np.amax(y)+20])
plt.show()


