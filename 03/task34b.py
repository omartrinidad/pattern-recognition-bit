import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def trainL2SVMPolyKernel(X, y, d, b=1., C=1., T=1000):
  m, n = X.shape
  I = np.eye(n)
  Y = np.outer(y,y)
  K = (b + np.dot(X.T, X))**d
  M = Y * K + Y + 1./C*I
  mu = np.ones(n) / n
  for t in range(T):
    eta = 2./(t+2)
    grd = 2 * np.dot(M, mu)
    mu += eta * (I[np.argmin(grd)] - mu)
  return mu

def applyL2SVMPolyKernel(x, XS, ys, ms, w0, d, b=1.):
  if x.ndim == 1:
    x = x.reshape(len(x),1)
  k = (b + np.dot(x.T, XS))**d
  return np.sum(k * ys * ms, axis=1) + w0

# load data
X = np.genfromtxt("data/xor-X.csv", dtype=float, delimiter=',')
Y = np.genfromtxt("data/xor-y.csv", dtype=float, delimiter=',')

m = trainL2SVMPolyKernel(X, Y, d=3, C=2., T=1000)
s = np.where(m>0)[0]
XS = X[:,s]
ys = Y[s]
ms = m[s]
w0 = np.dot(ys,ms)

plt.scatter(X[0,:], X[1,:], c = Y)

x = np.linspace(np.amin(X[0,:])-0.1, np.amax(X[0,:])+0.1, 1000)
y = np.linspace(np.amin(X[1,:])-0.1, np.amax(X[1,:])+0.1, 1000)
CX, CY = np.meshgrid(x, y)
zi = applyL2SVMPolyKernel(np.vstack((CX.ravel(),CY.ravel())), XS, ys, ms, w0, d=3)
zi = np.sign(zi).reshape((1000,1000))
cmap = colors.LinearSegmentedColormap.from_list("", ["blue","white","orange"])
plt.contourf(x,y,zi, alpha=0.2, levels=np.linspace(np.amin(zi.ravel()), np.amax(zi.ravel()), 101), cmap=cmap, antialiased = True)

plt.show()


from sklearn.svm import SVC
clf = SVC(kernel="poly", coef0=1, C=2)
clf.fit(X.T, Y)

plt.scatter(X[0,:], X[1,:], c = Y)

x = np.linspace(np.amin(X[0,:])-0.1, np.amax(X[0,:])+0.1, 1000)
y = np.linspace(np.amin(X[1,:])-0.1, np.amax(X[1,:])+0.1, 1000)
CX, CY = np.meshgrid(x, y)
zi = clf.predict(np.vstack((CX.ravel(),CY.ravel())).T).reshape((1000,1000))
cmap = colors.LinearSegmentedColormap.from_list("", ["blue","white","orange"])
plt.contourf(x,y,zi, alpha=0.2, levels=np.linspace(np.amin(zi.ravel()), np.amax(zi.ravel()), 101), cmap=cmap, antialiased = True)

plt.show()



