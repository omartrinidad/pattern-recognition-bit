import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

k = 2
X = np.genfromtxt("data/data-dimred-X.csv", dtype=float, delimiter=',').T
y = np.genfromtxt("data/data-dimred-y.csv", dtype=float, delimiter=',')

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#clf = LinearDiscriminantAnalysis(n_components=2)
#clf.fit(X, y)
#z = clf.transform(X)

#print(z.shape)
#plt.scatter(z[:,0], z[:,1], c=y)
#plt.savefig("pca_2d.png", bbox_inches="tight", pad_inches=0)
#plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(z[:,0], z[:,1], z[:,2], c=y);
#plt.savefig("pca_3d.png", bbox_inches="tight", pad_inches=0)
#plt.show()


n_examples = X.shape[0]
n_features = X.shape[1]
n_classes = 3

n_examples_class = np.zeros(n_classes)
for i in range(n_classes):
  n_examples_class[i] = y[np.where(y==i+1)].size

u = np.zeros((n_classes, n_features))
cov = np.zeros((n_classes, n_features, n_features))
for i in range(n_classes):
  indices = np.where(y==i+1)[0]
  u[i, :] = np.mean(X[indices, :], axis=0)
  cov[i, :, :] = np.cov(X[indices, :].T)

S_w = np.zeros((n_features, n_features))
for i in range(n_classes):
  S_w = S_w + cov[i, :, :]

u_all = np.mean(X, axis=0)

S_b = np.zeros((n_features, n_features))
for i in range(n_classes):
  S_b = S_b + np.outer(u[i,:]-u_all, u[i,:]-u_all)

C = np.dot(np.linalg.pinv(S_w), S_b)

eival, eivec = np.linalg.eigh(C)
eivec = eivec.T[::-1]
z = np.dot(X, eivec[:3].T)
z = z.T

plt.scatter(z[0], z[1], c=y)
plt.savefig("out/03/lda/lda_2d.png", bbox_inches="tight", pad_inches=0)
plt.show()

# plot in 2D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(z[0], z[1], z[2], c=y);
plt.savefig("out/03/lda/lda_3d.png", bbox_inches="tight", pad_inches=0)
plt.show()

