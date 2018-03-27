import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import time
from auxiliar import *

@save_figure()
def plotData(x, y, z, title, k=None, path="latex/knn1.tex"):

    xy = np.column_stack((x, y))
    print(xy.shape)

    fig, ax = plt.subplots()

    z = ['green' if i > 0 else 'red' for i in z]
    plt.plot(x, y, 1)

    ax.set_facecolor("#ffeeee")

    plt.legend(loc='upper left')
    plt.show()

    return plt


def column(matrix, i):
    return [float(row[i]) for row in matrix]


trainData = np.loadtxt('data/data2-train.dat',dtype=np.object,comments='#',delimiter=None)
testData = np.loadtxt('data/data2-test.dat',dtype=np.object,comments='#',delimiter=None)

#kNN(testData, trainData, 1)
#kNN(testData, trainData, 3)

k=3
x_train = np.asarray(column(trainData,0))
x_test = np.asarray(column(testData,0))

y_train = np.asarray(column(trainData,1))
y_test = np.asarray(column(testData,1))

labels_train = np.asarray(column(trainData,2))
labels_test = np.asarray(column(testData,2))

z_result = []

corr_rate = 0
total_sum = 0
start = time.time()

for i in range(x_test.shape[0]):
    x_i = x_test[i]
    y_i = y_test[i]
    tmp_list = []

    for j in range(x_train.shape[0]):
        x_j = x_train[j]
        y_j = y_train[j]
        label = labels_train[j]
        dist = sp.distance.euclidean([x_i,y_i],[x_j,y_j])
        tmp_list.append([dist,label])

    tmp_list = np.asarray(tmp_list)
    tmp_list = tmp_list[np.argsort(tmp_list[:,0])][0:k]
    result = 1.0 if sum(tmp_list[:,1]) > 0 else -1.0

    if labels_test[i] == result :
        corr_rate += 1.0
    total_sum += 1.0
    z_result.append(result)

end = time.time()
elapsed = end-start

#plotData(x_test, y_test, z_result,   "Classification results k={}".format(k), k=k, path="latex/knn{}.tex".format(k))
plotData(x_test, y_test, labels_test, "Test data k={}".format(k), k=k, path="latex/knn{}.tex".format(k))
