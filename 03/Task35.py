import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

# read data
data = np.loadtxt('whData.dat', dtype=np.object)

# Reading and storing columns 0 and 1 and finally converted to numpy.float array
data = data[:, 0:2].astype(np.float)

# Removing outliers
data = data[data[:, 0] > 0]

# create weight vector for train data
wgt = np.copy(data[:, 0])

# create height vector for train data
hgt = np.copy(data[:, 1])

xmin = hgt.min() - 15
xmax = hgt.max() + 15
ymin = wgt.min() - 15
ymax = wgt.max() + 15


def plot_data_and_fit(h, w, x, y):
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


def trsf(x):
    return x / 100.


n = 10
x = np.linspace(xmin, xmax, 100)

# method1:
# regression using ployfit
c = poly.polyfit(hgt, wgt, n)
y = poly.polyval(x, c)
plot_data_and_fit(hgt, wgt, x, y)

# method2:
# regression using the Vandermonde matrix and pinv
X = poly.polyvander(hgt, n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(x, n), c)
plot_data_and_fit(hgt, wgt, x, y)

# method3:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x, n), c)
plot_data_and_fit(hgt, wgt, x, y)

# method4:
# regression on transformed data using the Vandermonde
# matrix and either pinv or lstsq
X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x), n), c)
plot_data_and_fit(hgt, wgt, x, y)
